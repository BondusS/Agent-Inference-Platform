import os
import json
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]

class MCPConnection:
    """
    Client for Model Context Protocol (MCP) server.
    Handles service discovery of tools, context injection, and remote execution.
    """
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.tools: List[MCPTool] = []
        
    def discover_tools(self) -> List[MCPTool]:
        """Fetch available tools from the MCP server."""
        try:
            response = requests.get(f"{self.url}/tools", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.tools = [
                    MCPTool(
                        name=t["name"],
                        description=t["description"],
                        input_schema=t.get("input_schema", {})
                    )
                    for t in data.get("tools", [])
                ]
                return self.tools
        except Exception as e:
            print(f"Error discovering tools for MCP {self.name}: {e}")
        return []

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on the remote MCP server."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": f"tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                },
                "id": 1
            }
            response = requests.post(f"{self.url}/rpc", json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": f"Failed to execute tool {tool_name} on MCP {self.name}: {e}"}
        return {"error": "Unknown connection issue"}

class MCPRegistryManager:
    """
    Manages multiple MCP server connections.
    Integrates tools into LangChain/LangGraph agent routing lists.
    """
    def __init__(self, config_path: str = "./mcp_config.json"):
        self.config_path = config_path
        self.connections: Dict[str, MCPConnection] = {}
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = json.load(f)
                for item in config.get("servers", []):
                    self.add_server(item["name"], item["url"])
        else:
            # Seed default servers
            default_config = {
                "servers": [
                    {"name": "Google Search MCP", "url": "http://mcp-server-search:8001"},
                    {"name": "Weather MCP", "url": "http://mcp-server-weather:8002"}
                ]
            }
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=4)
            for item in default_config["servers"]:
                self.add_server(item["name"], item["url"])

    def add_server(self, name: str, url: str) -> MCPConnection:
        conn = MCPConnection(name, url)
        self.connections[name] = conn
        return conn

    def get_all_tools(self) -> List[Dict[str, Any]]:
        all_tools = []
        for name, conn in self.connections.items():
            # In production, we trigger discovery. Here we return structured metadata.
            tools = conn.tools if conn.tools else [
                MCPTool(name=f"{name.lower().replace(' ', '_')}_action", description=f"Execute action on {name}", input_schema={})
            ]
            for t in tools:
                all_tools.append({
                    "mcp_server": name,
                    "name": t.name,
                    "description": t.description,
                    "schema": t.input_schema
                })
        return all_tools
