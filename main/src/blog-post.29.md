# Model Context Protocol(MCP)
## Concept
![alt text](images/blog29_mcp_concept.png)

MCP is an open protocol that standardizes how applications provide context to LLMs. MCP provides a standardized way to connect AI models to different data sources and tools. 

 - MCP Hosts: Programs like Claude Desktop, IDEs, or AI tools that want to access data through MCP
 -MCP Clients: Protocol clients that maintain 1:1 connections with servers
 - MCP Servers: Lightweight programs that each expose specific capabilities through the standardized Model Context Protocol
 - Local Data Sources: Your computer’s files, databases, and services that MCP servers can securely access
- Remote Services: External systems available over the internet (e.g., through APIs) that MCP servers can connect to

 - https://www.anthropic.com/news/model-context-protocol
 - https://modelcontextprotocol.io/introduction

 ### How It is Implemented (Focusesd on tools)
 참고: https://github.com/modelcontextprotocol/python-sdk

MCP 서버와 클라이언트가 있을 때, 이 둘은 프로토콜로 각자의 리퀘스트와 리스폰스를 정의한다. 이 프로토콜은 Schema 이며 레퍼런스로 삼은 python-sdk 기준으로는 "pydantic"을 사용해서 validation을 한 Schema 이다.  

MCP 에서 빼놓을 수 없는 것이 각각의 tool (mcp-tool)의 호출인데, 이 부분을 중점으로 살펴본다.
이 "Tool"은 위에서 언급한 바와 같이 Schema(Protocol)로 정의 되어 있다.  

```python
  class Tool(BaseModel):
      name: str  # The name of the tool
      description: str | None = None  # Human-readable description
      inputSchema: dict[str, Any]  # JSON Schema for parameters
```
이 Schema를 먼저 클라이언트가 활용한다. 

```python
# mcp-client
all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )
```
클라이언트는 서버의 tool 리스트를 받아오고, tool의 description을 파싱한 정보를 이용해서 쿼리 질의용 메시지를 만든다.
이 질의를 통해서 LLM은 질의 내용에 따라서 어떤 Tool을 써야 할지, 어떻게 응답값을 주어야 할지를 알게 된다.

```python
# mcp-client
llm_response = self.llm_client.get_response(messages)
                    logging.info("\nAssistant: %s", llm_response)

                    result = await self.process_llm_response(llm_response)
```
그 후 클라이언트는 메시지를 이용해서 llm에 질의를 하고, 결과값을 ``process_llm_response`` 에 넘긴다.

```python
# mcp-client/process_llm_response
 try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )
...
 return llm_response
        except json.JSONDecodeError:
            return llm_response
```
사전에 llm에게 response 값의 포맷을 알려주었기 때문에 이 포캣을 파싱해서 tool 사용을 llm 이 요청을 하였는 지 확인할 수 있고, 요청 하였을 시 이를 실행하고 응답값을 다시 기존 컨텍스트에 넘겨주게 된다.

MCP 서버는 mcp-tools를 구현 해서 준비하는 것이 역할의 전부이다. 아래를 참고해보자.

```python
 app = Server("mcp-website-fetcher")

 @app.call_tool()
    async def fetch_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        if name != "fetch":
            raise ValueError(f"Unknown tool: {name}")
        if "url" not in arguments:
            raise ValueError("Missing required argument 'url'")
        return await fetch_website(arguments["url"])

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="fetch",
                description="Fetches a website and returns its content",
                inputSchema={
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch",
                        }
                    },
                },
            )
        ]
```
`` app = Server("mcp-simple-resource")`` 를 통해서 MCP Server 를 정의 한후, decorator를 활용해서 tool이 llm에 의해서 조회될 때와 (``@app.list_tools()``) 실제 tool이 실행될 때 (``@app.call_tool()``)의 핸들러를 각각 등록한다.

이제 클라이언트와 서버의 상호작용을 예시를 통해서 확인해보자.
먼저 위에서 본 바와 같이 클라이언트가 tool list를 서버에 요청하면 서버는 자신이 ``fetch`` 라는 tool을 가지고 있다고 응답하게 된다. 
```python
# Client sends:
{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": null
}

# Server responds:
{
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "tools": [
            {
                "name": "fetch",
                "description": "Fetches a website and returns its content",
                "inputSchema": {
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch"
                        }
                    }
                }
            },
            ...
        ]
    }
}
```
물론 여기서는 하나의 tool 뿐이지만, 실제로는 더 많은 tool들이 있을 것이다. 이 tool들의 정보는 ``system_message``를 구성하는 과정에서 잘 파싱되어 llm에게 전달되게 되고, llm 은 쿼리의 내용을 바탕으로 어떤 tool을 써야할지 결정 할 수 있는 근거를 가지게 된다.  
그리고 이후에 tool을 사용(call) 하는 경우는 아래와 같다.
```python
# (1) LLM output (parsed by client):
{
    "tool": "fetch",
    "arguments": {"url": "https://example.com"}
}

# (2) Client sends to server:
{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "fetch",
        "arguments": {"url": "https://example.com"}
    }
}
```
클라이언트는 "fetch" 라는 mcp-tool의 실행이 필요하다고 판단 후 (1) 와 같이 응답을 하게 되고, 이 응답값은 클라이언트에서 자체적으로 ``process_llm_response`` 함수를 거쳐 (2)의 스키마를 따라서 서버에 요청을 하게 된다. 이후 클라이언트는 mcp-tool를 사용한 서버의 응답값을 기존 컨텍스트에 넘겨주게 된다. 
