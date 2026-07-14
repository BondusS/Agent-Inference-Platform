import os
import json
import requests
from typing import List, Dict, Any
from pydantic import BaseModel
from langchain_core.tools import StructuredTool


class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPConnection:
    """Клиент для конкретного MCP сервера."""

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.tools: List[MCPTool] = []

    def discover_tools(self) -> List[MCPTool]:
        """Получает список доступных инструментов от MCP сервера."""
        try:
            # Делаем запрос к MCP серверу для получения списка тулов
            response = requests.get(f"{self.url}/tools", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.tools = [
                    MCPTool(
                        name=t.get("name", "unknown_tool"),
                        description=t.get("description", "No description"),
                        input_schema=t.get("input_schema", {})
                    )
                    for t in data.get("tools", [])
                ]
                return self.tools
        except Exception as e:
            print(f"Error discovering tools for MCP {self.name}: {e}")
        return []

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Отправляет JSON-RPC запрос на выполнение инструмента."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                },
                "id": 1
            }
            response = requests.post(f"{self.url}/rpc", json=payload, timeout=15)
            if response.status_code == 200:
                return json.dumps(response.json(), ensure_ascii=False)
            else:
                return f"MCP Error {response.status_code}: {response.text}"
        except Exception as e:
            return json.dumps({"error": f"Failed to execute {tool_name}: {e}"})


class MCPRegistryManager:
    """Управляет всеми MCP серверами и отдает готовые тулы для LangChain."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.connections: Dict[str, MCPConnection] = {}
        self.load_config()

    def load_config(self):
        """Загружает список серверов из mcps.json"""
        self.connections.clear()
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                try:
                    config = json.load(f)
                    # Формат mcps.json: {"Weather MCP": {"url": "...", "enabled": True}}
                    for name, data in config.items():
                        if data.get("enabled", True):
                            self.add_server(name, data["url"])
                except Exception as e:
                    print(f"Failed to parse mcps.json: {e}")

    def add_server(self, name: str, url: str) -> MCPConnection:
        conn = MCPConnection(name, url)
        conn.discover_tools()
        self.connections[name] = conn
        return conn

    def get_langchain_tools(self) -> List[StructuredTool]:
        """Преобразует найденные MCP тулы в формат StructuredTool для графа LangGraph"""
        lc_tools = []
        for conn_name, conn in self.connections.items():
            for mcp_tool in conn.tools:
                # Замыкание для создания уникальной функции под каждый тул
                def create_tool_function(connection: MCPConnection, tool_name: str):
                    def func(**kwargs: Any) -> str:
                        print(f"--> [LangGraph] Executing MCP Tool: {tool_name} with args {kwargs}")
                        return connection.execute_tool(tool_name, kwargs)

                    func.__name__ = tool_name
                    return func

                tool_func = create_tool_function(conn, mcp_tool.name)

                # Добавляем JSON Schema прямо в описание, чтобы модель понимала, какие аргументы передавать
                schema_str = json.dumps(mcp_tool.input_schema)
                full_description = f"[{conn_name}] {mcp_tool.description}. Arguments Schema: {schema_str}"

                lc_tools.append(StructuredTool.from_function(
                    func=tool_func,
                    name=mcp_tool.name,
                    description=full_description
                ))

        return lc_tools
