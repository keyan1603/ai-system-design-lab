"""Ticket-platform MCP server, built with Requisite's MCPServer.

Three read-only tools over the synthetic tables. Read-only is the point:
least-privilege tools are the real defense against a model being talked into
something harmful, because guardrails on text can always be bypassed but a tool
that cannot write cannot be abused to write.

Run standalone over stdio:  python -m tickets.mcp_server
"""

from requisite import MCPServer
from requisite.tools import tool

from tickets.data import CUSTOMERS, KB, ORDERS


@tool
def get_customer(customer_id: str) -> dict:
    """Look up a customer's plan, region and account status by id such as C-1001."""
    return CUSTOMERS.get(customer_id.strip().upper(), {"error": f"unknown customer {customer_id}"})


@tool
def get_order(order_id: str) -> dict:
    """Look up an order's amount, billing status and note by id such as O-5001."""
    return ORDERS.get(order_id.strip().upper(), {"error": f"unknown order {order_id}"})


@tool
def search_kb(topic: str) -> str:
    """Return the support knowledge-base article for a topic. Topics: password_reset, duplicate_charge, suspended_account, mobile_crash."""
    return KB.get(topic.strip().lower(), f"no article for topic '{topic}'")


server = MCPServer(name="ticket-ops", tools=[get_customer, get_order, search_kb])

if __name__ == "__main__":
    server.run_stdio()
