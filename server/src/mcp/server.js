import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// Import tools
import askFireExpertTool from "./tools/ask-fire-expert.tool.js";
import textifyDataTool from "./tools/textify-data.tool.js";
import getEmbeddingTool from "./tools/get-embedding.tool.js";
import healthCheckTool from "./tools/health-check.tool.js";

// ============================================================
// Tool Registry
// ============================================================
const TOOLS = [
  askFireExpertTool.definition,
  textifyDataTool.definition,
  getEmbeddingTool.definition,
  healthCheckTool.definition,
];

const HANDLERS = {
  ask_fire_expert: askFireExpertTool.handler,
  textify_data: textifyDataTool.handler,
  get_embedding: getEmbeddingTool.handler,
  health_check: healthCheckTool.handler,
};

// ============================================================
// MCP Server Setup
// ============================================================
const server = new Server(
  {
    name: "vfims-fire-investigation",
    version: "2.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    const handler = HANDLERS[name];
    if (!handler) {
      throw new Error(`Unknown tool: ${name}`);
    }

    const result = await handler(args || {});

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({ error: error.message }, null, 2),
        },
      ],
      isError: true,
    };
  }
});

// ============================================================
// Start Server (when run directly)
// ============================================================
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("VFIMS MCP Server started (stdio transport)");
}

// Export for use as module
export { server, TOOLS, HANDLERS };

// Run if this is the main module
main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
