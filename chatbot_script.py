"""
Core script for the Enhanced Syllabus RAG Chatbot.

This module defines the main chatbot class `EnhancedSyllabusRAGChatbot`
and the setup function `setup_enhanced_chatbot` to initialize and configure it.
The chatbot uses a Retrieval-Augmented Generation (RAG) approach with
Google's Gemini model, SentenceTransformers for embeddings, and ChromaDB
for vector storage.
"""
import json
import re
import os
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import warnings

# Load environment variables from a .env file
load_dotenv()

warnings.filterwarnings('ignore')


class EnhancedSyllabusRAGChatbot:
    """
    A RAG-based chatbot for answering questions about university syllabi.

    This class encapsulates all the logic for loading data, creating a vector store,
    retrieving relevant context, and generating responses using a generative AI model.
    """

    def __init__(self, gemini_api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the chatbot components.

        Args:
            gemini_api_key (str): The API key for the Google Gemini model.
            model_name (str): The name of the Gemini model to use.
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./chroma_db_by_dept")
        self.collection = None

        # Data stores
        self.chunks_data = []
        self.faq_data = []
        self.concept_mapping = {}
        self.course_details = {}
        self.semester_course_map = {}

    def load_data(self, syllabus_data_path: str, optimization_path: str):
        """
        Loads and processes syllabus and optimization data from JSON files.

        Args:
            syllabus_data_path (str): Path to the main syllabus data file.
            optimization_path (str): Path to the RAG optimization data file (e.g., FAQs).
        """
        with open(syllabus_data_path, 'r', encoding='utf-8') as f:
            restructured_data = json.load(f)

        self.course_details = {course['metadata']['course_code']: course for course in restructured_data}

        for course in restructured_data:
            metadata = course.get('metadata', {})
            course_code = metadata.get('course_code')
            course_name = metadata.get('course_name')
            semester = metadata.get('semester')
            if not all([course_code, course_name, semester]):
                continue

            semester = str(semester)
            if semester not in self.semester_course_map:
                self.semester_course_map[semester] = []

            self.semester_course_map[semester].append({
                "course_code": course_code,
                "course_name": course_name,
                "credits": metadata.get('credits', 'N/A'),
                "category": metadata.get('category', 'N/A')
            })

            # Create an overview chunk for each course
            overview_content = (f"Course Overview for {course_name} ({course_code}): "
                                f"This is a Semester {semester} '{metadata.get('category', 'N/A')}' course with {metadata.get('credits', 'N/A')} credits. "
                                f"Prerequisites: {metadata.get('prerequisites', 'Not specified')}.")
            self.chunks_data.append({'content': overview_content, 'metadata': metadata, 'chunk_type': 'overview'})
            
            # --- FIX: Create a dedicated chunk for course outcomes ---
            outcomes = metadata.get('course_outcomes', [])
            if outcomes and isinstance(outcomes, list):
                outcomes_content = f"The course outcomes for {course_name} ({course_code}) are: {'; '.join(outcomes)}"
                self.chunks_data.append({'content': outcomes_content, 'metadata': metadata, 'chunk_type': 'outcomes'})
            # --- END OF FIX ---


            # Create a chunk for each syllabus unit
            for unit in course.get('syllabus', []):
                 if not isinstance(unit, dict):
                     print(f"Warning: Skipping malformed syllabus unit for course {course_code}. Unit data: {unit}")
                     continue

                 unit_topics = unit.get('topics', 'Not specified')
                 if isinstance(unit_topics, list):
                     unit_topics = ", ".join(unit_topics)
                 unit_content = (f"Syllabus for {course_name} ({course_code}), Unit {unit.get('unit_number', '')} "
                                 f"titled '{unit.get('title', 'N/A')}': {unit_topics}")
                 self.chunks_data.append({'content': unit_content, 'metadata': metadata, 'chunk_type': 'syllabus_unit'})

            # Create a chunk for textbooks and references
            books = course.get('books', {})
            textbooks = books.get('textbooks', [])
            ref_books = books.get('reference_books', [])
            if textbooks or ref_books:
                books_content = (f"Reading materials for {course_name} ({course_code}). "
                                 f"Textbooks: {', '.join(textbooks) if textbooks else 'None listed'}. "
                                 f"Reference Books: {', '.join(ref_books) if ref_books else 'None listed'}.")
                self.chunks_data.append({'content': books_content, 'metadata': metadata, 'chunk_type': 'books'})

        # Load optimization data (FAQs, etc.)
        try:
            with open(optimization_path, 'r', encoding='utf-8') as f:
                optimization_data = json.load(f)
                self.faq_data = optimization_data.get('faq_dataset', [])
                self.concept_mapping = optimization_data.get('concept_mapping', {})
        except FileNotFoundError:
            print(f"Warning: Optimization file not found at {optimization_path}. Running without it.")
            self.faq_data = []
            self.concept_mapping = {}

        print(f"Loaded data for {len(self.course_details)} courses and {len(self.faq_data)} FAQ entries.")

    def create_enhanced_vector_store(self, collection_name: str):
        """
        Creates a fresh ChromaDB vector store for the loaded data.
        It will delete any existing collection with the same name to ensure
        data is always up-to-date on server restart.

        Args:
            collection_name (str): The unique name for the ChromaDB collection.
        """
        try:
            print(f"Checking for and deleting existing collection '{collection_name}' to refresh data.")
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Info: Could not delete collection '{collection_name}' (it may not exist yet).")
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        documents, metadatas, ids = [], [], []

        # Process syllabus chunks
        for i, chunk in enumerate(self.chunks_data):
            documents.append(chunk['content'])
            metadata_cleaned = {k: str(v) for k, v in chunk['metadata'].items() if v is not None}
            metadata_cleaned['chunk_type'] = chunk.get('chunk_type', 'unknown')
            metadata_cleaned['source'] = 'syllabus'
            metadatas.append(metadata_cleaned)
            ids.append(f"chunk_{i}")

        # Process semester summary chunks
        for semester, courses in self.semester_course_map.items():
            course_list_str = "\n".join([f"- {c['course_code']}: {c['course_name']} ({c['credits']} credits, {c['category']})" for c in courses])
            semester_content = f"The following courses are offered in Semester {semester}:\n{course_list_str}"
            documents.append(semester_content)
            metadatas.append({'semester': semester, 'chunk_type': 'semester_summary', 'source': 'semester_map', 'total_courses': str(len(courses))})
            ids.append(f"semester_{semester}")

        # Process FAQ chunks
        for i, faq in enumerate(self.faq_data):
            documents.append(f"Question: {faq['question']} Answer: {faq['answer']}")
            metadatas.append({'category': faq.get('category', 'general'), 'chunk_type': 'faq', 'source': 'faq'})
            ids.append(f"faq_{i}")

        # Add all documents to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch_docs).tolist()
            self.collection.add(embeddings=embeddings, documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
        print(f"Successfully created and populated vector store '{collection_name}' with {len(documents)} documents.")


    def enhance_query(self, query: str) -> str:
        """
        Enhances the user query with related terms for better retrieval.

        Args:
            query (str): The original user query.

        Returns:
            str: The enhanced query.
        """
        enhanced_terms = []
        query_lower = query.lower()
        # Enhance with course names if a semester is mentioned
        sem_match = re.search(r'\bsem(?:ester)?\s*(\d+)\b', query_lower)
        if sem_match:
            sem_num = sem_match.group(1)
            if sem_num in self.semester_course_map:
                course_names = [c['course_name'] for c in self.semester_course_map[sem_num]]
                enhanced_terms.extend(course_names)
        # Enhance with course name if a course code is mentioned
        course_code_pattern = re.search(r'\b\d{2}[A-Z&]{2,}\d{4}[A-Z]?\b', query.upper())
        if course_code_pattern:
            course_code = course_code_pattern.group()
            if course_code in self.course_details:
                course_name = self.course_details[course_code]['metadata'].get('course_name', '')
                enhanced_terms.append(course_name)
        # Enhance based on predefined concept mappings
        for concept, courses in self.concept_mapping.items():
            if concept.lower() in query_lower:
                enhanced_terms.extend(courses[:2])
        return f"{query} {' '.join(enhanced_terms)}" if enhanced_terms else query

    def retrieve_context(self, query: str, n_results: int = 8) -> List[Dict]:
        """
        Retrieves relevant context documents from the vector store.

        Args:
            query (str): The user's query.
            n_results (int): The number of context documents to retrieve.

        Returns:
            List[Dict]: A list of retrieved documents with metadata and distance scores.
        """
        if not self.collection:
            raise ValueError("Vector store not initialized.")
        enhanced_query = self.enhance_query(query)
        query_embedding = self.embedding_model.encode([enhanced_query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        return [{'content': doc, 'metadata': meta, 'distance': dist}
                for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])]

    def generate_enhanced_response(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generates a final response using the Gemini model based on the retrieved context.

        Args:
            query (str): The user's query.
            context_docs (List[Dict]): The context documents retrieved from the vector store.

        Returns:
            str: The generated answer.
        """
        context_parts = [
            f"Context Snippet (Source: {doc['metadata'].get('source', 'unknown')}, Course: {doc['metadata'].get('course_code', 'N/A')}):\n{doc['content']}"
            for doc in context_docs
        ]
        context_text = "\n---\n".join(context_parts)
        
        # Default prompt for general queries
        prompt = f"""You are a precise and helpful academic assistant for a university syllabus. Your task is to answer the student's question concisely based ONLY on the provided context.

        Context Information:
        ---
        {context_text}
        ---

        Student's Question: {query}

        Instructions:
        1. Synthesize a coherent, friendly answer from the provided context. Do not just list the raw snippets.
        2. If the context contains a list of courses, present them clearly in a list format.
        3. If specific details like textbooks, prerequisites, or outcomes are available in the context, integrate them naturally into your response.
        4. If the information to answer the question is NOT in the context, you MUST explicitly state that you cannot find the information in the provided documents.
        5. Be direct and clear in your response.

        Answer:"""

        # If the user specifically asks for the "syllabus", use a more focused prompt.
        query_lower = query.lower()
        if 'syllabus' in query_lower and not any(keyword in query_lower for keyword in ['book', 'overview', 'credit', 'outcome', 'prerequisite']):
            prompt = f"""You are a precise academic assistant. A student is asking specifically for the syllabus for a course. Based ONLY on the provided context, answer their question.

            Context Information:
            ---
            {context_text}
            ---

            Student's Question: {query}

            Instructions:
            1. Find the syllabus units and topics in the context.
            2. List ONLY the syllabus units with their topics. Format them clearly with unit numbers and titles.
            3. CRITICAL: Do NOT include the Course Overview, prerequisites, course outcomes, credits, or book recommendations in your answer.
            4. If the syllabus unit information is not available in the context, explicitly state that.

            Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_message = str(e)
            if "API key not valid" in error_message:
                return "Sorry, there is an issue with the server's API key configuration. Please contact the administrator."
            return f"Sorry, I encountered an error generating the response: {error_message}"

    def chat(self, query: str, n_context: int = 10) -> Dict[str, Any]:
        """
        Orchestrates the chat process from query to response.
        """
        context_docs = self.retrieve_context(query, n_context)
        response = self.generate_enhanced_response(query, context_docs)
        relevant_courses = sorted(list(set(
            f"{doc['metadata'].get('course_code')} - {doc['metadata'].get('course_name')}"
            for doc in context_docs if doc['metadata'].get('course_code')
        )))
        return {
            'query': query,
            'answer': response,
            'context_used': len(context_docs),
            'relevant_courses': relevant_courses
        }

def setup_enhanced_chatbot(gemini_api_key: str, department: str, regulation: Optional[str] = None, data_root: str = "data"):
    """
    Factory function to initialize and set up a chatbot instance.
    """
    if not gemini_api_key:
        raise ValueError("Gemini API key is required.")
        
    if regulation:
        data_path = os.path.join(data_root, department, regulation)
        collection_name = f"syllabus_collection_{department}_{regulation}"
    else:
        data_path = os.path.join(data_root, department)
        collection_name = f"syllabus_collection_{department}"

    syllabus_data_path = os.path.join(data_path, "syllabus_data.json")
    optimization_path = os.path.join(data_path, "rag_optimization_data.json")

    if not os.path.exists(syllabus_data_path):
        raise FileNotFoundError(f"Syllabus data not found for '{department}' (Regulation: {regulation or 'N/A'}) at {syllabus_data_path}")

    chatbot = EnhancedSyllabusRAGChatbot(gemini_api_key)
    chatbot.load_data(syllabus_data_path, optimization_path)
    chatbot.create_enhanced_vector_store(collection_name=collection_name)
    return chatbot

def run_test_query(chatbot: EnhancedSyllabusRAGChatbot, query: str):
    """Helper function to run a single test query and print the result."""
    print(f"\n[Query]: {query}")
    response = chatbot.chat(query)
    print(f"[Response]: {response['answer']}")
    print(f"  (Context Docs Used: {response['context_used']}, Relevant Courses: {len(response['relevant_courses'])})")

def main():
    """Main function to run standalone tests on the chatbot script."""
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
        print("ERROR: Please set your GEMINI_API_KEY in the .env file for testing.")
        return

    print("--- Initializing Chatbot Tests ---")
    try:
        print("\n--- Testing CE - VR23 Regulation ---")
        ce_vr23_chatbot = setup_enhanced_chatbot(API_KEY, 'ce', regulation='vr23')
        run_test_query(ce_vr23_chatbot, "What are the outcomes for 23BS1101?")

        print("\n--- Testing CE - SU24 Regulation ---")
        ce_su24_chatbot = setup_enhanced_chatbot(API_KEY, 'ce', regulation='su24')
        run_test_query(ce_su24_chatbot, "what are the subjects in sem 3")

    except FileNotFoundError as e:
        print(f"\nERROR during setup: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

