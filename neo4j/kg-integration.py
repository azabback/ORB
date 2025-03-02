import os
from dotenv import load_dotenv
import google.generativeai as genai
from neo4j import GraphDatabase, RoutingControl

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = "neo4j"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class App:
    def __init__(self, uri, user, password, database, gemini_key):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.database = database
        
        system_instructions = """You are a text entity extractor. Your role is to parse the given text,
        identity all of the entities in the text, and return a list of all unique entities.
        The entities should be unique. A single entity can appear in different formats in the text, so if any two conceptually mean the same thing, make sure no duplicates appear.
        Provide one list with unique items with the header "Unique Entities".
        Provide another list with the header "Duplicate Entities" and this format: (kept entity: omitted duplicate entitie with commas).
        Do not provide extra information. Only provide a comma separated lists without spaces around commas."""
        
        genai.configure(api_key=gemini_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite", system_instruction=system_instructions)
        # response_text = self.model.generate_content(prompt).text

    def close(self):
        self.driver.close()
    
    def test_llm(self):
        sample_text = "I wonder what the entities in this paragraph are. Yo, I have a question. It's about BLE encryption. And tesla cars and how they use ble. BLE stands for bluetooth low energy. Do you know that? Encryption is cool."
        response_text = self.model.generate_content(sample_text).text
        print(response_text)
    
    def test_entities_query(self, entities):
        query = """
                MATCH (n)-[r]-(m)
                WHERE ANY(entity IN $entities WHERE 
                    toLower(n.id) CONTAINS toLower(entity) OR 
                    toLower(m.id) CONTAINS toLower(entity))
                RETURN n.id, r, m.id
                """
        eager_result = self.driver.execute_query(
            query, entities=entities, database_=self.database, routing_=RoutingControl.READ
        )
        return eager_result

    def test_query(self):
        query = (
            "MATCH (n)-[r]-(m) "
            "WHERE n.id CONTAINS 'BLE' AND m.id CONTAINS 'BLE' "
            "RETURN n.id, r, m.id"
        )
        eager_result = self.driver.execute_query(
            query, database_=self.database, routing_=RoutingControl.READ
            # result_transformer_=lambda r: {
            #     "node_id": r["n.id"], 
            #     "relationship": r["r"].type 
            # }
        )
        return eager_result

    def display_eager_result(eager_result):
        records = eager_result[0]
        for record in records:
            node_n_id = record["n.id"] 
            relationship = record["r"].type
            node_m_id = record["m.id"] 

            if relationship != "HAS_ENTITY":
                # print(f"(Node n ID: {node_n_id}, Relationship: {relationship}, Node m ID: {node_m_id})")
                print(f"({node_n_id} {relationship} {node_m_id})")

if __name__ == "__main__":
    app = App(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, GEMINI_API_KEY)

    try:
        # eager_result = app.test_query()
        # app.display_eager_result(eager_result)
        
        # print("#################################")
        # entities = ["ble", "PKES", "ABC"]
        # eager_result = app.test_entities_query(entities)
        # app.display_eager_result(eager_result)

        app.test_llm()
        
    finally:
        app.close()
