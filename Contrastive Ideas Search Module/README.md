Contrastive Ideas Search Module

**Description:**  
This module identifies documents or strings with **semantically opposing ideas** (e.g., "AI improves healthcare" vs. "AI harms healthcare"), not just dissimilar content. It combines **topic relevance** and **stance opposition detection** using a hybrid embedding strategy. The module leverages a vector database and advanced NLP techniques to ensure retrieved documents are both topically related and semantically conflicting.

---

### **Key Features & Requirements:**

#### **1. Hybrid Embedding Strategy**
   - **Topic Embedding**:  
     Uses a standard embedding model (e.g., `mxbai-embed-large`) to ensure retrieved documents share the **same topic** as the input.
   - **Stance Embedding**:  
     Uses a fine-tuned model to detect **semantic opposition** (e.g., trained on contradiction datasets like SNLI or Debatepedia).

#### **2. Two-Phase Search Pipeline**
   1. **Topic Filtering**:  
      Retrieve documents with high topic similarity to narrow down candidates.
   2. **Stance Analysis**:  
      Rank filtered documents by their stance opposition score.

#### **3. Contrastive Scoring**
   - **Final Score**:  
     `final_score = (topic_similarity) * (1 + stance_opposition)`  
     Prioritizes documents that are both topically related and semantically opposed.
   - **Stance Opposition**:  
     Computed via dot product of stance embeddings, inverted to maximize contrast.

#### **4. Vector Database Schema**
   ```python
   class Document:
       id: str
       text: str
       topic_embedding: List[float]  # For topic matching
       stance_embedding: List[float]  # For opposition detection
       metadata: Dict  # Source, timestamp, etc.
   ```
   - **Database**: Chroma DB or Pinecone with support for multi-vector indexing.

#### **5. Training Requirements**
   - **Stance Model**:  
     Fine-tune using triplet loss on contradiction datasets:
     ```python
     from sentence_transformers import SentenceTransformer, losses
     model = SentenceTransformer("mxbai-embed-large")
     loss = losses.ContrastiveLoss(model)  # Anchor vs. Positive (contrast) pairs
     ```
   - **Training Data**:  
     Use labeled contradiction pairs (e.g., ["Coffee is healthy", "Coffee is unhealthy"]).

#### **6. Edge Case Handling**
   - **Unrelated Documents**: Discard candidates with low topic similarity (`topic_similarity < threshold`).
   - **Ambiguity**: Apply confidence thresholds (`stance_opposition > 0.5`) to filter weak contrasts.

---

### **Workflow**

1. **Input**: Document/string to analyze (e.g., "AI improves healthcare").
2. **Embedding Generation**:
   - Generate `topic_embedding` and `stance_embedding` for the input.
3. **Topic Filtering**:
   - Query vector DB for top `N` documents with highest `topic_similarity`.
4. **Stance Analysis**:
   - Compute `stance_opposition` for topic-filtered documents.
   - Calculate `final_score` and rank results.
5. **Output**: Top `k` documents with the most contrastive ideas.

---

### **Implementation Guidelines**

1. **Stance-Aware Model**:
   - Fine-tune on contradiction datasets using `sentence-transformers`.
   - Use triplet loss to separate opposing pairs from unrelated ones.

2. **Vector Database**:
   - Use a database supporting multi-vector search (e.g., Pinecone, Weaviate).
   - Index both `topic_embedding` and `stance_embedding`.

3. **Hybrid Query**:
   ```python
   results = vector_db.query(
       vector={
           "topic_embedding": query_topic_emb,
           "stance_embedding": query_stance_emb
       },
       strategy="hybrid",
       weights=[0.3, 0.7]  # Weight stance opposition higher
   )
   ```

4. **Thresholds**:
   - Set `topic_similarity_threshold = 0.6` (configurable).
   - Set `stance_opposition_threshold = 0.5`.

---

### **Example Use Case**

**Input**:  
"Renewable energy can fully replace fossil fuels by 2030."

**Output**:
1. "Renewable energy lacks the scalability to replace fossil fuels before 2050."  
   (Topic similarity: 0.85, Stance opposition: 0.92 → **Final score: 1.63**)
2. "Fossil fuels are irreplaceable due to energy density requirements."  
   (Topic similarity: 0.78, Stance opposition: 0.88 → **Final score: 1.49**)
3. "Nuclear energy is the only viable replacement for fossil fuels."  
   (Topic similarity: 0.65, Stance opposition: 0.45 → **Discarded: stance < 0.5**)

---

### **Acceptance Criteria**
1. Fully functional two-phase search pipeline with hybrid scoring.
2. Fine-tuned stance-aware embedding model.
3. Vector DB schema supporting multi-vector storage/querying.
4. Configurable thresholds for topic similarity and stance opposition.
5. Unit tests covering edge cases (e.g., ambiguous or unrelated documents).
6. Documentation with API examples and training guidelines.
