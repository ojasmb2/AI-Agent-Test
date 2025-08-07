import sqlite3
from mcp.server.fastmcp import FastMCP

# Create FastMCP instance
mcp = FastMCP("SQLExecutor")

# Connect to SQLite database
conn = sqlite3.connect("sakila.db")

@mcp.tool()
def execute_query(query: str) -> list:
    """Executes a SQL query and returns the results"""
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()

# ✅ Start the FastMCP server
if __name__ == "__main__":
    print("✅ Starting MCP server on http://localhost:5001 ...")
    mcp.run()
