import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from neo4j import GraphDatabase, RoutingControl

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = "neo4j"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DISPLAY_ALL = True

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
        Do not provide extra information. Only provide a comma separated lists without spaces around commas.
        Do not use new lines for spacing. There should only be two lines in the response (unique and duplicates).
        Entity extraction from my text should be thorough without any conceptually duplicate entities."""
        
        genai.configure(api_key=gemini_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite", system_instruction=system_instructions)

    def close(self):
        self.driver.close()

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

    def test_llm_entity_extraction(self):
        sample_text = "I wonder what the entities in this paragraph are. Yo, I have a question. It's about BLE encryption. And tesla cars and how they use ble. BLE stands for bluetooth low energy. Do you know that? Encryption is cool."
        entities_response_text = self.model.generate_content(sample_text).text
        return self.parse_entities_response(entities_response_text)
    
    ###############################################################

    def retrieve_facts_from_text(self, text):
        if DISPLAY_ALL: print("TEXT: ", text)
        entities_response_text = self.model.generate_content(text).text
        entities = self.parse_entities_response(entities_response_text)
        if DISPLAY_ALL: print("ENTITIES: ",entities)
        return self.query_entities(entities)

    def parse_entities_response(self, response_text):
        match = re.search(r"Unique Entities:(.*?)\n", response_text)
        if match:
            unique_entities = set(map(str.strip, match.group(1).split(',')))
            return list(unique_entities)
        return []

    def query_entities(self, entities: list):
        query = """
                MATCH (n)-[r]-(m)
                WHERE ANY(entity IN $entities WHERE 
                    toLower(n.id) CONTAINS toLower(entity))
                    AND type(r) <> "HAS_ENTITY"
                RETURN n.id, r, m.id
                """
        eager_result = self.driver.execute_query(
            query, entities=entities, database_=self.database, routing_=RoutingControl.READ
        )
        return eager_result

    def display_result(self, eager_result):
        records = eager_result[0]
        print("#"*20)
        for record in records:
            node_n_id = record["n.id"] 
            relationship = record["r"].type
            node_m_id = record["m.id"] 
            print(f"({node_n_id} {relationship} {node_m_id})")
        print("#"*20)
        print(f"Returned {len(records)} results")


if __name__ == "__main__":
    app = App(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, GEMINI_API_KEY)

    try:
        # eager_result = app.test_query()
        # app.display_result(eager_result)
        
        # print("#################################")
        # entities = ["ble", "PKES", "ABC"]
        # eager_result = app.query_entities(entities)
        # app.display_result(eager_result)

        # print(app.test_llm_entity_extraction())


        # example #1

        # test the factual completeness of the llm response 
        # user prompt -> entities in prompt -> all relevant facts -> validate if most are present in llm response
        # IMPORTANT NOTE: not as viable since user prompts may not contain queryable entities or entities incorrect
        # example_user_prompt = "Provide a brief summary of this research paper"
        # app.display_result(app.retrieve_facts_from_text(example_user_prompt))

        # test the factual accuracy of the llm response itself
        # llm response -> entities in response -> facts -> validate if statements match to one or more facts
        # example_llm_response = "This research paper examines security vulnerabilities in Tesla Model 3's keyless entry system. It analyzes the pairing and authentication protocols for both Key Cards and Phone Keys. The researchers discovered that weaknesses exist that allow unauthorized access, including man-in-the-middle attacks via Bluetooth. They developed an application to demonstrate these vulnerabilities, showing how a Tesla Model 3 can be unlocked and driven without the owner's knowledge. The paper underscores flaws in the Tesla keyless entry system, and proposes methods for mitigating these risks. The paper aims to highlight insecure implementations of Bluetooth in security-critical systems."
        # app.display_result(app.retrieve_facts_from_text(example_llm_response))

        # example #2
        # example_user_prompt = "How do the identified vulnerabilities in Tesla key systems impact real-world security?"
        # res = app.retrieve_facts_from_text(example_user_prompt)
        # if DISPLAY_ALL: app.display_result(res)

        example_llm_response = "The vulnerabilities in Tesla key systems have real-world security impacts, including unauthorized access and theft. Key Cards lacking certificate verification can be replaced with easily duplicated, compromised third-party cards. Phone Keys are susceptible to relay attacks due to the lack of Bluetooth Low Energy (BLE) encryption and static MAC addresses, which can be exploited via improved MitM attacks. The token value used in Phone Key authentication remains fixed for extended periods, creating opportunities for exploitation. These attacks can occur without the owner's knowledge, potentially leading to vehicle theft. While PIN to Drive can prevent driving, it does not prevent unauthorized entry"
        res = app.retrieve_facts_from_text(example_llm_response)
        if DISPLAY_ALL: app.display_result(res)
        
        
    finally:
        app.close()
