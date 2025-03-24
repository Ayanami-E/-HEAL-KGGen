from config import logger

def extract_medical_entities(text, kg_handler):
    """Extract medical entities from text"""
    entities = []
    
    # Dictionary-based simple entity matching
    if kg_handler:
        for node_id, node in kg_handler.nodes.items():
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

def retrieve_knowledge_subgraph(entities, kg_handler, max_hops=1):
    """Retrieve knowledge subgraph related to entities"""
    if not kg_handler:
        return {"entities": [], "relationships": []}
    
    subgraph_info = {
        "entities": [],
        "relationships": []
    }
    
    processed_nodes = set()
    processed_edges = set()
    
    for entity in entities:
        # Add entity itself
        if entity["id"] not in processed_nodes:
            node = kg_handler.get_node_by_id(entity["id"])
            if node:
                subgraph_info["entities"].append(node)
                processed_nodes.add(entity["id"])
        
        # Get related relationships
        relationships = kg_handler.get_relationships_for_node(entity["id"])
        for rel in relationships:
            rel_id = f"{rel.get('source')}_{rel.get('type')}_{rel.get('target')}"
            if rel_id not in processed_edges:
                subgraph_info["relationships"].append(rel)
                processed_edges.add(rel_id)
                
                # Add node at the other end of relationship
                other_node_id = rel.get("target") if rel.get("source") == entity["id"] else rel.get("source")
                if other_node_id not in processed_nodes:
                    other_node = kg_handler.get_node_by_id(other_node_id)
                    if other_node:
                        subgraph_info["entities"].append(other_node)
                        processed_nodes.add(other_node_id)
    
    logger.info(f"Retrieved related knowledge subgraph: {len(subgraph_info['entities'])} entities, {len(subgraph_info['relationships'])} relationships")
    return subgraph_info

def format_knowledge_as_context(subgraph_info):
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

def format_entity_highlights(entities):
    """Format extracted entities for display in the prompt"""
    if not entities:
        return "No relevant medical entities identified."
    
    result = ""
    entities_by_category = {}
    
    # Group entities by category
    for entity in entities:
        category = entity.get("category", "unknown")
        if category not in entities_by_category:
            entities_by_category[category] = []
        entities_by_category[category].append(entity)
    
    # Format entity information by category
    for category, category_entities in entities_by_category.items():
        if category_entities:
            result += f"\n{category.capitalize()}:\n"
            for entity in category_entities:
                result += f"- {entity.get('name', '')}\n"
    
    return result