import json
from config import logger

class JSONKnowledgeGraphHandler:
    """Processor for handling knowledge graph data in JSON format"""
    
    def __init__(self, json_path):
        """Initialize JSON knowledge graph handler"""
        self.json_path = json_path
        self.data = None
        self.nodes = {}  # Nodes indexed by ID
        self.edges = []  # List of relationships
        self.node_categories = {}  # Nodes indexed by category
        self._load_json()
        self._index_graph()
        
    def _load_json(self):
        """Load knowledge graph from JSON file"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Successfully loaded data: {self.json_path}")
        except Exception as e:
            logger.error(f"{e}")
            self.data = {"nodes": [], "edges": []}
            
    def _index_graph(self):
        """Create indexes for nodes and edges for quick querying"""
        # Process nodes
        if "nodes" in self.data:
            for node in self.data["nodes"]:
                node_id = node.get("id", "")
                if node_id:
                    self.nodes[node_id] = node
                    
                    # Index by category
                    category = node.get("category", "unknown")
                    if category not in self.node_categories:
                        self.node_categories[category] = []
                    self.node_categories[category].append(node)
        
        # Process edges
        if "edges" in self.data:
            self.edges = self.data["edges"]
            
        logger.info(f" {len(self.nodes)}  , {len(self.edges)} ")
        
    def get_node_count(self):
        """Get total number of nodes in the knowledge graph"""
        return len(self.nodes)
        
    def get_relationship_count(self):
        """Get total number of relationships in the knowledge graph"""
        return len(self.edges)
        
    def get_nodes_by_category(self, category):
        """Get all nodes of a specific category"""
        return self.node_categories.get(category, [])
        
    def get_node_by_id(self, node_id):
        """Get a specific node by ID"""
        return self.nodes.get(node_id)
        
    def get_relationships_by_type(self, rel_type):
        """Get all relationships of a specific type"""
        return [edge for edge in self.edges if edge.get("type") == rel_type]
        
    def get_relationships_for_node(self, node_id):
        """Get all relationships for a specific node"""
        relationships = []
        for edge in self.edges:
            if edge.get("source") == node_id or edge.get("target") == node_id:
                relationships.append(edge)
        return relationships
        
    def search_nodes(self, query, categories=None):
        """Search for nodes in the knowledge graph"""
        results = []
        
        for node_id, node in self.nodes.items():
            # If categories specified, only check those categories
            if categories and node.get("category") not in categories:
                continue
                
            # Simple text matching
            if query.lower() in node_id.lower():
                results.append(node)
                
        return results
        
    def query_disease_info(self, disease_name):
        """Query information for a specific disease"""
        # Find matching disease nodes
        disease_nodes = []
        for node_id, node in self.nodes.items():
            if node.get("category") == "disease" and disease_name.lower() in node_id.lower():
                disease_nodes.append(node)
                
        if not disease_nodes:
            return {"message": f"Disease not found: {disease_name}"}
            
        # Collect related information
        result = []
        for disease_node in disease_nodes:
            disease_id = disease_node.get("id")
            disease_info = {
                "disease": disease_id,
                "symptoms": [],
                "treatments": [],
                "biomarkers": []
            }
            
            # Get related relationships
            relationships = self.get_relationships_for_node(disease_id)
            for rel in relationships:
                rel_type = rel.get("type", "")
                source = rel.get("source", "")
                target = rel.get("target", "")
                
                # Determine related node (not the disease node)
                other_node_id = target if source == disease_id else source
                other_node = self.get_node_by_id(other_node_id)
                
                if not other_node:
                    continue
                    
                other_category = other_node.get("category", "")
                
                # Categorize information
                if other_category == "symptom":
                    disease_info["symptoms"].append(other_node_id)
                elif other_category == "treatment":
                    disease_info["treatments"].append(other_node_id)
                elif other_category in ["gene", "biomarker", "protein"]:
                    disease_info["biomarkers"].append(other_node_id)
                    
            result.append(disease_info)
            
        return result
        
    def find_genetic_biomarkers(self, disease_name):
        """Find gene biomarkers related to a disease"""
        biomarkers = []
        
        # Find matching disease nodes
        disease_nodes = []
        for node_id, node in self.nodes.items():
            if node.get("category") == "disease" and disease_name.lower() in node_id.lower():
                disease_nodes.append(node)
                
        if not disease_nodes:
            return biomarkers
            
        # Find related biomarkers
        for disease_node in disease_nodes:
            disease_id = disease_node.get("id")
            
            # Get direct relationships
            relationships = self.get_relationships_for_node(disease_id)
            
            for rel in relationships:
                source = rel.get("source", "")
                target = rel.get("target", "")
                
                # Determine related node
                other_node_id = target if source == disease_id else source
                other_node = self.get_node_by_id(other_node_id)
                
                if not other_node:
                    continue
                    
                other_category = other_node.get("category", "")
                
                # Add potential biomarkers
                if other_category in ["gene", "biomarker", "protein"]:
                    biomarker = {
                        "id": other_node_id,
                        "name": other_node_id.split(": ")[-1] if ": " in other_node_id else other_node_id,
                        "category": other_category,
                        "relationship": rel.get("type", "associated")
                    }
                    
                    # Avoid duplicates
                    if not any(b["id"] == biomarker["id"] for b in biomarkers):
                        biomarkers.append(biomarker)
                        
        return biomarkers
        
    def extract_medical_entities(self, text):
        """Extract medical entities from text"""
        entities = []
        
        # Dictionary-based simple entity matching
        for node_id, node in self.nodes.items():
            entity_name = node_id.split(": ")[-1] if ": " in node_id else node_id
            # Use more precise matching to avoid short word mismatches
            if len(entity_name) > 3 and entity_name.lower() in text.lower():
                entities.append({
                    "id": node_id,
                    "name": entity_name,
                    "category": node.get("category", "unknown")
                })
        
        # Filter out duplicate entities, keep longer matches
        filtered_entities = []
        entity_names = set()
        for entity in sorted(entities, key=lambda x: len(x["name"]), reverse=True):
            if entity["name"].lower() not in entity_names:
                filtered_entities.append(entity)
                entity_names.add(entity["name"].lower())
        
        logger.info(f"Extracted {len(filtered_entities)} medical entities")
        return filtered_entities
        
    def retrieve_knowledge_subgraph(self, entities, max_hops=1):
        """Retrieve knowledge subgraph related to entities"""
        subgraph_info = {
            "entities": [],
            "relationships": []
        }
        
        processed_nodes = set()
        processed_edges = set()
        
        for entity in entities:
            # Add entity itself
            if entity["id"] not in processed_nodes:
                node = self.get_node_by_id(entity["id"])
                if node:
                    subgraph_info["entities"].append(node)
                    processed_nodes.add(entity["id"])
            
            # Get related relationships
            relationships = self.get_relationships_for_node(entity["id"])
            for rel in relationships:
                rel_id = f"{rel.get('source')}_{rel.get('type')}_{rel.get('target')}"
                if rel_id not in processed_edges:
                    subgraph_info["relationships"].append(rel)
                    processed_edges.add(rel_id)
                    
                    # Add node at the other end of relationship
                    other_node_id = rel.get("target") if rel.get("source") == entity["id"] else rel.get("source")
                    if other_node_id not in processed_nodes:
                        other_node = self.get_node_by_id(other_node_id)
                        if other_node:
                            subgraph_info["entities"].append(other_node)
                            processed_nodes.add(other_node_id)
        
        logger.info(f"Retrieved related knowledge subgraph: {len(subgraph_info['entities'])} entities, {len(subgraph_info['relationships'])} relationships")
        return subgraph_info
        
    def format_knowledge_as_context(self, subgraph_info):
        """Convert knowledge subgraph to text context"""
        if not subgraph_info["entities"] and not subgraph_info["relationships"]:
            return ""
        
        context = "Related medical knowledge:\n"
        
        # Organize entities by category
        entities_by_category = {}
        for entity in subgraph_info["entities"]:
            category = entity.get("category", "unknown")
            if category not in entities_by_category:
                entities_by_category[category] = []
            entities_by_category[category].append(entity)
        
        # Add entity information
        for category, entities in entities_by_category.items():
            if entities:
                context += f"\n{category.capitalize()}:\n"
                for entity in entities:
                    entity_name = entity.get("id", "").split(": ")[-1] if ": " in entity.get("id", "") else entity.get("id", "")
                    if not entity_name:
                        continue
                    properties = ", ".join([f"{k}: {v}" for k, v in entity.items() if k not in ["id", "category"] and v])
                    context += f"- {entity_name}" + (f" ({properties})" if properties else "") + "\n"
        
        # Add relationship information
        if subgraph_info["relationships"]:
            context += "\nRelationships:\n"
            for rel in subgraph_info["relationships"]:
                source = rel.get("source", "").split(": ")[-1] if ": " in rel.get("source", "") else rel.get("source", "")
                target = rel.get("target", "").split(": ")[-1] if ": " in rel.get("target", "") else rel.get("target", "")
                rel_type = rel.get("type", "")
                if source and target and rel_type:
                    context += f"- {source} --[{rel_type}]--> {target}\n"
        
        logger.info(f"Generated knowledge context: {len(context.split())} words")
        return context