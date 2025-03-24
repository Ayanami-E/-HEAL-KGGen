from config import logger

def enhance_question_with_kg(question, options, kg_handler):
    """
    Enhance a medical question with knowledge graph information to improve LLM accuracy
    
    Args:
        question: The medical question text
        options: Dictionary of answer options (A, B, C, D)
        kg_handler: Knowledge graph handler instance
    
    Returns:
        Enhanced prompt for the LLM with KG-extracted information
    """
    # Extract entities from question with more aggressive matching
    question_entities = kg_handler.extract_medical_entities(question)
    logger.info(f"Extracted {len(question_entities)} entities from question")
    
    # Extract entities from all options with more aggressive matching
    options_text = ""
    for key, value in options.items():
        options_text += f"{key}: {value} "
    
    options_entities = kg_handler.extract_medical_entities(options_text)
    logger.info(f"Extracted {len(options_entities)} entities from options")
    
    # Combine all unique entities
    all_entities = []
    entity_ids = set()
    
    # Prioritize question entities
    for entity in question_entities:
        if entity["id"] not in entity_ids:
            all_entities.append(entity)
            entity_ids.add(entity["id"])
    
    # Then add option entities
    for entity in options_entities:
        if entity["id"] not in entity_ids:
            all_entities.append(entity)
            entity_ids.add(entity["id"])
    
    # Expand entities to include related important entities (2 hops for key entities)
    expanded_entities = expand_entities_with_related(all_entities, kg_handler)
    
    # Retrieve knowledge subgraph for expanded entities
    subgraph = kg_handler.retrieve_knowledge_subgraph(expanded_entities, max_hops=2)
    
    # Format knowledge as context
    knowledge_context = kg_handler.format_knowledge_as_context(subgraph)
    
    # Analyze options to identify distinguishing facts
    option_analysis = analyze_options_with_kg(options, expanded_entities, kg_handler)
    
    # Create enhanced prompt with KG information
    options_formatted = ""
    for key, value in options.items():
        options_formatted += f"{key}: {value}\n"
    
    enhanced_prompt = f"""
Question: {question}

Options:
{options_formatted}

Key medical entities identified in this question and options:
{format_entity_highlights(expanded_entities)}

{knowledge_context}

Option Analysis:
{option_analysis}

Based on the knowledge graph information above, determine which option contains the correct gene-disease associations.
Analyze each option carefully for consistency with the knowledge graph data.
Your answer must be a single letter (A, B, C, or D).
"""
    
    return enhanced_prompt

def expand_entities_with_related(entities, kg_handler):
    """Expand the entity list with closely related entities that might be relevant"""
    expanded = entities.copy()
    existing_ids = {e["id"] for e in entities}
    
    priority_categories = ["gene", "disease", "biomarker", "protein"]
    priority_relationship_types = ["associated_with", "causes", "treats", "biomarker_of", "regulates"]
    
    # First expansion: Add directly connected high-value entities
    for entity in entities:
        # Only expand from priority categories
        if entity.get("category") not in priority_categories:
            continue
            
        # Get relationships for this entity
        relationships = kg_handler.get_relationships_for_node(entity["id"])
        
        # Filter for priority relationships
        for rel in relationships:
            rel_type = rel.get("type", "")
            
            # Skip non-priority relationship types
            if rel_type not in priority_relationship_types:
                continue
                
            # Get the entity at the other end
            source_id = rel.get("source")
            target_id = rel.get("target")
            other_id = target_id if source_id == entity["id"] else source_id
            
            # Skip if already included
            if other_id in existing_ids:
                continue
                
            # Get the entity
            other_node = kg_handler.get_node_by_id(other_id)
            if not other_node:
                continue
                
            # Add to expanded list if it's a priority category
            other_category = other_node.get("category", "")
            if other_category in priority_categories:
                expanded.append({
                    "id": other_id,
                    "name": other_id.split(": ")[-1] if ": " in other_id else other_id,
                    "category": other_category
                })
                existing_ids.add(other_id)
    
    logger.info(f"Expanded entities from {len(entities)} to {len(expanded)}")
    return expanded

def analyze_options_with_kg(options, entities, kg_handler):
    """Analyze each option for its relationship to the knowledge graph"""
    analysis = ""
    
    for key, option_text in options.items():
        # Extract entities from this option
        option_entities = kg_handler.extract_medical_entities(option_text)
        entity_names = [e["name"] for e in option_entities]
        
        # Count relationships between entities in this option
        relationship_count = count_relationships_between_entities(option_entities, kg_handler)
        
        # Check for gene-disease pairs mentioned in the option
        gene_disease_pairs = find_gene_disease_pairs(option_entities, kg_handler)
        
        # Summarize findings for this option
        analysis += f"Option {key}: Contains {len(option_entities)} medical entities"
        if entity_names:
            analysis += f" including {', '.join(entity_names[:3])}"
            if len(entity_names) > 3:
                analysis += f" and {len(entity_names)-3} more"
        analysis += f". Found {relationship_count} relationships between these entities in the knowledge graph.\n"
        
        # Add gene-disease information if present
        if gene_disease_pairs:
            analysis += f"  Gene-Disease relationships: {', '.join(gene_disease_pairs[:3])}"
            if len(gene_disease_pairs) > 3:
                analysis += f" and {len(gene_disease_pairs)-3} more"
            analysis += "\n"
    
    return analysis

def count_relationships_between_entities(entities, kg_handler):
    """Count how many relationships exist between the given entities"""
    count = 0
    entity_ids = [e["id"] for e in entities]
    
    for i, entity1_id in enumerate(entity_ids):
        for entity2_id in entity_ids[i+1:]:
            # Check if there's a direct relationship
            relationships = kg_handler.get_relationships_for_node(entity1_id)
            for rel in relationships:
                if rel.get("source") == entity1_id and rel.get("target") == entity2_id:
                    count += 1
                elif rel.get("source") == entity2_id and rel.get("target") == entity1_id:
                    count += 1
    
    return count

def find_gene_disease_pairs(entities, kg_handler):
    """Find gene-disease pairs mentioned in the entities"""
    pairs = []
    
    # Separate genes and diseases
    genes = [e for e in entities if e.get("category") == "gene"]
    diseases = [e for e in entities if e.get("category") == "disease"]
    
    # Check for relationships between genes and diseases
    for gene in genes:
        for disease in diseases:
            # Get all relationships for the gene
            gene_relationships = kg_handler.get_relationships_for_node(gene["id"])
            
            # Check if any relationship connects to the disease
            for rel in gene_relationships:
                if (rel.get("source") == gene["id"] and rel.get("target") == disease["id"]) or \
                   (rel.get("source") == disease["id"] and rel.get("target") == gene["id"]):
                    # Add this pair
                    rel_type = rel.get("type", "associated_with")
                    pairs.append(f"{gene['name']} {rel_type} {disease['name']}")
    
    return pairs

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

def create_kg_enhanced_prompt_system():
    """Create a system prompt that emphasizes KG integration"""
    return """You are a medical genetics expert with access to a specialized knowledge graph.
Your task is to analyze gene-disease questions using the knowledge provided.

Guidelines:
1. Pay special attention to the entities and relationships extracted from the knowledge graph
2. Focus on the correct gene-disease associations mentioned in the knowledge context
3. Look for direct relationships between genes and diseases in the provided knowledge
4. When seeing relationships like "Gene A --[associated_with]--> Disease B", consider this strong evidence
5. Pay attention to the Option Analysis section which highlights relationships found in each option
6. Prioritize information from the knowledge graph over general knowledge
7. Analyze each option carefully for consistency with the knowledge graph data
8. Return ONLY a single letter (A, B, C, or D) as your final answer with no explanation
"""

def ask_kg_enhanced_question(question, options, kg_handler, llm_interface):
    """Ask a question with KG enhancement"""
    # Create enhanced prompt
    enhanced_prompt = enhance_question_with_kg(question, options, kg_handler)
    
    # Create system prompt
    system_prompt = create_kg_enhanced_prompt_system()
    
    # Use a lower temperature for more deterministic answers
    response = llm_interface.ask(enhanced_prompt, system_prompt=system_prompt, temperature=0.1)
    
    # Extract letter answer
    answer_letter = llm_interface.get_single_letter_answer(response)
    
    if not answer_letter:
        logger.warning("Could not extract answer letter, defaulting to 'A'")
        answer_letter = "A"
    
    return answer_letter, enhanced_prompt