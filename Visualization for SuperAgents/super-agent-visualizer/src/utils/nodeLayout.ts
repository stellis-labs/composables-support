import ELK from "elkjs";
import { Node, Edge } from "reactflow";

// Initialize ELK with default configurations
const elk = new ELK({
  defaultLayoutOptions: {
    'elk.algorithm': 'layered',  // Options: layered, force, radial, etc.
    'elk.direction': 'DOWN',     // Layout direction
    'elk.layered.spacing.nodeNodeBetweenLayers': '100',
    'elk.spacing.nodeNode': '80', // Spacing between nodes
  },
});

interface ElkNode {
  id: string;
  x?: number;
  y?: number;
  children?: ElkNode[];
}

/**
 * Auto-layout function to position ReactFlow nodes using ELK.js
 */
const nodeLayout = async (nodes: Node[], edges: Edge[]) => {
  // Convert ReactFlow nodes to ELK format
  const elkNodes: ElkNode[] = nodes.map(node => ({
    id: node.id,
    width: node.width || 250,
    height: Math.max(180, node.data?.label?.length || 0 / 5),
  }));

  // Convert ReactFlow edges to ELK format
  const elkEdges = edges.map(edge => ({
    id: edge.id,
    sources: [edge.source],
    targets: [edge.target],
  }));

  // Define the graph structure
  const graph = {
    id: "root",
    children: elkNodes,
    edges: elkEdges,
  };

  try {
    // Apply ELK layout algorithm
    const layoutedGraph = await elk.layout(graph);
    console.log("hi1111:layout Graoph", layoutedGraph);
    // Map updated positions back to ReactFlow format
    return nodes.map(node => {
      const layoutNode = layoutedGraph.children?.find(n => n.id === node.id);
      return {
        ...node,
        position: layoutNode
          ? { x: layoutNode.x || 0, y: layoutNode.y || 0 }
          : { x: 0, y: 0 },
      };
    });
  } catch (error) {
    console.error("ELK layout error:", error);
    return nodes; // Return original nodes if layout fails
  }
};

export default nodeLayout;
