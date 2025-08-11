import sqlite3
from mcp.server.fastmcp import FastMCP

# Create FastMCP instance
mcp = FastMCP("SQLExecutor")

# Connect to SQLite database with thread safety
conn = sqlite3.connect("sakila.db", check_same_thread=False)

@mcp.tool()
def execute_query(query: str) -> list:
    """Executes a SQL query and returns the results"""
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()

if __name__ == "__main__":
    print("✅ Starting MCP server on http://localhost:5001 …")
    mcp.run()