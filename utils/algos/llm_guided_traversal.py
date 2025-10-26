#!/usr/bin/env python3
"""
LLM-Guided Traversal Algorithm
===============================

Uses an LLM to make traversal decisions at each hop, allowing semantic reasoning
beyond pure cosine similarity. The LLM acts as an agent that chooses which chunk
to explore next based on the query and current context.

This is a proof-of-concept for agentic RAG using LLM reasoning to guide retrieval.
"""

import time
import json
from typing import List, Dict, Any, Set, Optional
from .base_algorithm import BaseRetrievalAlgorithm, RetrievalResult
from ..traversal import TraversalPath, GranularityLevel, ConnectionType


class LLMGuidedTraversalAlgorithm(BaseRetrievalAlgorithm):
    """Algorithm: LLM-guided graph traversal using reasoning instead of pure similarity."""

    def __init__(self, knowledge_graph, config: Dict[str, Any],
                 query_similarity_cache: Dict[str, float], logger=None,
                 shared_embedding_model=None, llm_client=None):
        super().__init__(knowledge_graph, config, query_similarity_cache, logger, shared_embedding_model)

        # LLM-specific parameters
        self.top_k_candidates = self.traversal_config.get('llm_top_k_candidates', 5)
        self.llm_client = llm_client
        self.llm_model = self.traversal_config.get('llm_model', 'llama3.2:3b')
        self.llm_temperature = self.traversal_config.get('llm_temperature', 0.1)

        # Token tracking for cost analysis
        self.total_llm_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self.logger.info(f"LLMGuidedTraversalAlgorithm initialized: max_hops={self.max_hops}, "
                        f"top_k_candidates={self.top_k_candidates}, llm_model={self.llm_model}")

    def _initialize_llm_client(self):
        """Initialize Ollama client if not provided."""
        if self.llm_client is None:
            try:
                import ollama
                self.llm_client = ollama
                self.logger.info(f"✅ Initialized Ollama client for model: {self.llm_model}")
            except ImportError:
                raise ImportError(
                    "Ollama package not found. Install with: pip install ollama"
                )

    def _create_llm_prompt(self, query: str, extracted_sentences: List[str],
                          candidates: List[Dict[str, Any]]) -> str:
        """
        Create a structured prompt for the LLM to choose next chunk.

        Args:
            query: Original user query
            extracted_sentences: Sentences extracted so far
            candidates: List of candidate chunks with metadata

        Returns:
            Formatted prompt string
        """
        # Summarize extracted content (limit to avoid huge prompts)
        if extracted_sentences:
            extracted_summary = " ".join(extracted_sentences[:5])
            if len(extracted_summary) > 500:
                extracted_summary = extracted_summary[:500] + "..."
            extraction_info = f"ALREADY EXTRACTED ({len(extracted_sentences)} sentences):\n{extracted_summary}\n\n"
        else:
            extraction_info = "ALREADY EXTRACTED: None (starting traversal)\n\n"

        # Format candidates
        candidates_text = "CANDIDATE CHUNKS (pick ONE or STOP):\n"
        for i, candidate in enumerate(candidates, 1):
            chunk_preview = candidate['preview']
            if len(chunk_preview) > 200:
                chunk_preview = chunk_preview[:200] + "..."
            candidates_text += f"{i}. [{candidate['chunk_id']}] (similarity: {candidate['similarity']:.3f})\n"
            candidates_text += f"   Preview: {chunk_preview}\n\n"

        prompt = f"""You are a knowledge graph traversal agent. Your goal: find relevant content to answer the query.

QUERY: {query}

{extraction_info}{candidates_text}
INSTRUCTIONS:
- Choose the chunk number (1-{len(candidates)}) that seems most relevant to answering the query
- If you believe we have enough information to answer the query, respond with "stop"
- Consider both what we've already extracted and what new information each candidate provides
- Respond ONLY with a JSON object in this exact format:

{{"choice": <number 1-{len(candidates)} OR "stop">, "reasoning": "brief explanation"}}

Your response:"""

        return prompt

    def _parse_llm_response(self, response: str, candidates: List[Dict[str, Any]]) -> Optional[str]:
        """
        Parse LLM response to extract chunk choice.

        Args:
            response: Raw LLM response text
            candidates: List of candidate chunks

        Returns:
            chunk_id to traverse to, or "stop" to end traversal, or None on error
        """
        try:
            # Try to parse as JSON
            response_clean = response.strip()

            # Handle code blocks if present
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0].strip()

            parsed = json.loads(response_clean)
            choice = parsed.get('choice')
            reasoning = parsed.get('reasoning', 'No reasoning provided')

            self.logger.info(f"   LLM reasoning: {reasoning}")

            # Handle "stop" decision
            if isinstance(choice, str) and choice.lower() == "stop":
                self.logger.info("   LLM decided to stop traversal")
                return "stop"

            # Handle numeric choice
            if isinstance(choice, int) and 1 <= choice <= len(candidates):
                chosen_chunk = candidates[choice - 1]['chunk_id']
                self.logger.info(f"   LLM chose chunk {choice}: {chosen_chunk}")
                return chosen_chunk

            # Invalid choice
            self.logger.warning(f"   Invalid LLM choice: {choice}")
            return None

        except json.JSONDecodeError as e:
            self.logger.warning(f"   Failed to parse LLM response as JSON: {e}")
            self.logger.debug(f"   Raw response: {response[:200]}...")
            return None
        except Exception as e:
            self.logger.error(f"   Error parsing LLM response: {e}")
            return None

    def _ask_llm_for_next_chunk(self, query: str, extracted_sentences: List[str],
                               candidates: List[Dict[str, Any]]) -> Optional[str]:
        """
        Ask LLM to choose the next chunk to explore.

        Returns:
            chunk_id to explore, "stop" to end, or None on error
        """
        self._initialize_llm_client()

        # Create prompt
        prompt = self._create_llm_prompt(query, extracted_sentences, candidates)

        # Estimate prompt tokens (rough approximation: ~4 chars per token)
        prompt_tokens = len(prompt) // 4
        self.total_prompt_tokens += prompt_tokens

        try:
            self.logger.debug(f"   Calling LLM with {len(candidates)} candidates...")

            # Call Ollama
            response = self.llm_client.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': self.llm_temperature,
                    'num_predict': 150  # Limit response length
                }
            )

            self.total_llm_calls += 1

            # Extract response text
            response_text = response.get('response', '')

            # Estimate completion tokens
            completion_tokens = len(response_text) // 4
            self.total_completion_tokens += completion_tokens

            self.logger.debug(f"   LLM response ({completion_tokens} tokens): {response_text[:100]}...")

            # Parse response
            return self._parse_llm_response(response_text, candidates)

        except Exception as e:
            self.logger.error(f"   LLM call failed: {e}")
            return None

    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Execute LLM-guided traversal retrieval.

        Args:
            query: The search query
            anchor_chunk: Starting chunk for traversal

        Returns:
            RetrievalResult with LLM-guided traversal results
        """
        start_time = time.time()

        self.logger.info(f"🤖 LLMGuidedTraversal: Starting from anchor {anchor_chunk}")

        # Reset token counters
        self.total_llm_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Traversal state
        current_chunk = anchor_chunk
        visited_chunks: Set[str] = {anchor_chunk}
        extracted_sentences: List[str] = []
        path_nodes = [anchor_chunk]
        connection_types = []
        granularity_levels = [GranularityLevel.CHUNK]
        hop_count = 0
        llm_stopped = False

        # Extract sentences from anchor chunk
        anchor_sentences = self.get_chunk_sentences(anchor_chunk)
        extracted_sentences.extend(anchor_sentences)

        self.logger.info(f"   Extracted {len(anchor_sentences)} sentences from anchor")

        # Main traversal loop
        while len(extracted_sentences) < self.min_sentence_threshold and hop_count < self.max_hops:
            hop_count += 1

            self.logger.debug(f"🤖 Hop {hop_count}: Processing chunk {current_chunk}")

            # Extract sentences from current chunk (skip anchor)
            if hop_count > 1:
                chunk_sentences = self.get_chunk_sentences(current_chunk)
                newly_extracted = self.deduplicate_sentences(chunk_sentences, extracted_sentences)
                extracted_sentences.extend(newly_extracted)

                self.logger.info(f"📦 EXTRACTED: {len(newly_extracted)} new sentences from {current_chunk}")

            # Get top-k candidate chunks based on query similarity
            candidate_chunks = self._get_top_k_candidates(current_chunk, visited_chunks)

            if not candidate_chunks:
                self.logger.debug(f"   No candidate chunks found")
                break

            # Ask LLM to choose next chunk
            llm_choice = self._ask_llm_for_next_chunk(query, extracted_sentences, candidate_chunks)

            if llm_choice is None:
                # LLM failed - fall back to highest similarity
                self.logger.warning("   LLM failed, falling back to similarity-based choice")
                next_chunk = candidate_chunks[0]['chunk_id']
            elif llm_choice == "stop":
                # LLM decided we have enough information
                llm_stopped = True
                self.logger.info(f"🎯 LLM STOPPING: Decided we have enough information ({len(extracted_sentences)} sentences)")
                break
            else:
                next_chunk = llm_choice

            # TRAVERSE to chosen chunk
            if next_chunk not in visited_chunks:
                self.logger.info(f"🚶 TRAVERSE: Moving to LLM-chosen chunk {next_chunk}")
                current_chunk = next_chunk
                visited_chunks.add(next_chunk)
                path_nodes.append(next_chunk)
                connection_types.append(ConnectionType.RAW_SIMILARITY)
                granularity_levels.append(GranularityLevel.CHUNK)
            else:
                self.logger.debug(f"   Chosen chunk {next_chunk} already visited")
                break

        # Finalize results
        final_sentences = extracted_sentences[:self.max_results]
        confidence_scores = self.calculate_confidence_scores(final_sentences)
        sentence_sources = self.create_sentence_sources_mapping(final_sentences)

        # Create traversal path
        traversal_path = TraversalPath(
            nodes=path_nodes,
            connection_types=connection_types,
            granularity_levels=granularity_levels,
            total_hops=len(connection_types),
            is_valid=True,
            validation_errors=[]
        )

        # Calculate final score
        final_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        processing_time = time.time() - start_time

        self.logger.info(f"✅ LLMGuidedTraversal completed: {len(final_sentences)} sentences, "
                        f"{hop_count} hops in {processing_time:.3f}s")
        self.logger.info(f"   LLM stats: {self.total_llm_calls} calls, "
                        f"~{self.total_prompt_tokens + self.total_completion_tokens} total tokens")

        return RetrievalResult(
            algorithm_name="LLMGuidedTraversal",
            traversal_path=traversal_path,
            retrieved_content=final_sentences,
            confidence_scores=confidence_scores,
            query=query,
            total_hops=traversal_path.total_hops,
            final_score=final_score,
            processing_time=processing_time,
            metadata={
                'anchor_chunk': anchor_chunk,
                'hops_completed': hop_count,
                'chunks_visited': len(visited_chunks),
                'extraction_strategy': 'llm_guided_reasoning',
                'llm_stopped': llm_stopped,
                'llm_model': self.llm_model,
                'llm_calls': self.total_llm_calls,
                'estimated_prompt_tokens': self.total_prompt_tokens,
                'estimated_completion_tokens': self.total_completion_tokens,
                'estimated_total_tokens': self.total_prompt_tokens + self.total_completion_tokens
            },
            extraction_metadata={
                'total_extracted': len(extracted_sentences),
                'final_count': len(final_sentences),
                'extraction_points': len([node for node, granularity in zip(path_nodes, granularity_levels)
                                        if granularity == GranularityLevel.CHUNK])
            },
            sentence_sources=sentence_sources,
            query_similarities={sent: self.query_similarity_cache.get(self._find_sentence_id(sent), 0.0)
                              for sent in final_sentences}
        )

    def _get_top_k_candidates(self, current_chunk: str, visited_chunks: Set[str]) -> List[Dict[str, Any]]:
        """
        Get top-k candidate chunks based on query similarity.

        Returns:
            List of candidate dictionaries with chunk_id, similarity, and preview
        """
        # Get connected chunks
        chunk_obj = self.kg.chunks.get(current_chunk)
        if not chunk_obj:
            return []

        # Get all connected chunks (intra + inter document)
        connected_chunks = chunk_obj.intra_doc_connections + chunk_obj.inter_doc_connections

        # Filter out visited chunks and score by query similarity
        candidates = []
        for chunk_id in connected_chunks:
            if chunk_id in visited_chunks:
                continue

            # Get query similarity from cache
            similarity = self.query_similarity_cache.get(chunk_id, 0.0)

            # Get chunk preview
            chunk_text = self._get_chunk_text(chunk_id)

            candidates.append({
                'chunk_id': chunk_id,
                'similarity': similarity,
                'preview': chunk_text
            })

        # Sort by query similarity (highest first) and take top-k
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:self.top_k_candidates]

    def _get_chunk_text(self, chunk_id: str) -> str:
        """Get the text content of a chunk."""
        chunk_obj = self.kg.chunks.get(chunk_id)
        if not chunk_obj:
            return ""

        if hasattr(chunk_obj, 'chunk_text'):
            return chunk_obj.chunk_text

        # Fallback: combine sentences
        sentences = self.get_chunk_sentences(chunk_id)
        return " ".join(sentences) if sentences else ""
