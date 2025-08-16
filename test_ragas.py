#!/usr/bin/env python3
"""
RAGAS Test Script - Minimal Working Example
==========================================

This script tests RAGAS question generation to understand exactly what inputs it expects.
Run this to see the structure RAGAS needs for successful question generation.
"""

import os
import json
from typing import List, Dict, Any

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "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"


def create_dummy_documents():
    """Create simple test documents that should work with RAGAS."""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="""
            Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalise to unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance. ML finds application in many fields, including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine. The application of ML to business problems is known as predictive analytics. Statistics and mathematical optimisation (mathematical programming) methods comprise the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) via unsupervised learning. From a theoretical viewpoint, probably approximately correct learning provides a framework for describing machine learning. The term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and pioneer in the field of computer gaming and artificial intelligence. The synonym self-teaching computers was also used in this time period. The earliest machine learning program was introduced in the 1950s when Arthur Samuel invented a computer program that calculated the winning chance in checkers for each side, but the history of machine learning roots back to decades of human desire and effort to study human cognitive processes. In 1949, Canadian psychologist Donald Hebb published the book The Organization of Behavior, in which he introduced a theoretical neural structure formed by certain interactions among nerve cells. Hebb's model of neurons interacting with one another set a groundwork for how AIs and machine learning algorithms work under nodes, or artificial neurons used by computers to communicate data. Other researchers who have studied human cognitive systems contributed to the modern machine learning technologies as well, including logician Walter Pitts and Warren McCulloch, who proposed the early mathematical models of neural networks to come up with algorithms that mirror human thought processes. By the early 1960s, an experimental \"learning machine\" with punched tape memory, called Cybertron, had been developed by Raytheon Company to analyse sonar signals, electrocardiograms, and speech patterns using rudimentary reinforcement learning. It was repetitively \"trained\" by a human operator/teacher to recognise patterns and equipped with a \"goof\" button to cause it to reevaluate incorrect decisions. A representative book on research into machine learning during the 1960s was Nilsson's book on Learning Machines, dealing mostly with machine learning for pattern classification. Interest related to pattern recognition continued into the 1970s, as described by Duda and Hart in 1973. In 1981 a report was given on using teaching strategies so that an artificial neural network learns to recognise 40 characters (26 letters, 10 digits, and 4 special symbols) from a computer terminal. Tom M. Mitchell provided a widely quoted, more formal definition of the algorithms studied in the machine learning field: \"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.\" This definition of the tasks in which machine learning is concerned offers a fundamentally operational definition rather than defining the field in cognitive terms. This follows Alan Turing's proposal in his paper \"Computing Machinery and Intelligence\", in which the question \"Can machines think?\" is replaced with the question \"Can machines do what we (as thinking entities) can do?\". Modern-day machine learning has two objectives. One is to classify data based on models which have been developed; the other purpose is to make predictions for future outcomes based on these models. A hypothetical algorithm specific to classifying data may use computer vision of moles coupled with supervised learning in order to train it to classify the cancerous moles. A machine learning algorithm for stock trading may inform the trader of future potential predictions. As a scientific endeavour, machine learning grew out of the quest for artificial intelligence (AI). In the early days of AI as an academic discipline, some researchers were interested in having machines learn from data. They attempted to approach the problem with various symbolic methods, as well as what were then termed \"neural networks\"; these were mostly perceptrons and other models that were later found to be reinventions of the generalised linear models of statistics. Probabilistic reasoning was also employed, especially in automated medical diagnosis. However, an increasing emphasis on the logical, knowledge-based approach caused a rift between AI and machine learning. Probabilistic systems were plagued by theoretical and practical problems of data acquisition and representation. By 1980, expert systems had come to dominate AI, and statistics was out of favour. Work on symbolic/knowledge-based learning did continue within AI, leading to inductive logic programming(ILP), but the more statistical line of research was now outside the field of AI proper, in pattern recognition and information retrieval. Neural networks research had been abandoned by AI and computer science around the same time. This line, too, was continued outside the AI/CS field, as \"connectionism\", by researchers from other disciplines including John Hopfield, David Rumelhart, and Geoffrey Hinton. Their main success came in the mid-1980s with the reinvention of backpropagation. Machine learning (ML), reorganised and recognised as its own field, started to flourish in the 1990s. The field changed its goal from achieving artificial intelligence to tackling solvable problems of a practical nature. It shifted focus away from the symbolic approaches it had inherited from AI, and toward methods and models borrowed from statistics, fuzzy logic, and probability theory. Machine learning and data mining often employ the same methods and overlap significantly, but while machine learning focuses on prediction, based on known properties learned from the training data, data mining focuses on the discovery of (previously) unknown properties in the data (this is the analysis step of knowledge discovery in databases). Data mining uses many machine learning methods, but with different goals; on the other hand, machine learning also employs data mining methods as \"unsupervised learning\" or as a preprocessing step to improve learner accuracy. Much of the confusion between these two research communities (which do often have separate conferences and separate journals, ECML PKDD being a major exception) comes from the basic assumptions they work with: in machine learning, performance is usually evaluated with respect to the ability to reproduce known knowledge, while in knowledge discovery and data mining (KDD) the key task is the discovery of previously unknown knowledge. Evaluated with respect to known knowledge, an uninformed (unsupervised) method will easily be outperformed by other supervised methods, while in a typical KDD task, supervised methods cannot be used due to the unavailability of training data. Machine learning also has intimate ties to optimisation: Many learning problems are formulated as minimisation of some loss function on a training set of examples. Loss functions express the discrepancy between the predictions of the model being trained and the actual problem instances (for example, in classification, one wants to assign a label to instances, and models are trained to correctly predict the preassigned labels of a set of examples). Characterizing the generalisation of various learning algorithms is an active topic of current research, especially for deep learning algorithms. Machine learning and statistics are closely related fields in terms of methods, but distinct in their principal goal: statistics draws population inferences from a sample, while machine learning finds generalisable predictive patterns. According to Michael I. Jordan, the ideas of machine learning, from methodological principles to theoretical tools, have had a long pre-history in statistics. He also suggested the term data science as a placeholder to call the overall field. Conventional statistical analyses require the a priori selection of a model most suitable for the study data set. In addition, only significant or theoretically relevant variables based on previous experience are included for analysis.
            """,
            metadata={"source": "ml_basics.txt", "title": "Machine Learning Basics"}
        ),
        Document(
            page_content="""
            In contrast, machine learning is not built on a pre-structured model; rather, the data shape the model by detecting underlying patterns. The more variables (input) used to train the model, the more accurate the ultimate model will be. Leo Breiman distinguished two statistical modelling paradigms: data model and algorithmic model, wherein \"algorithmic model\" means more or less the machine learning algorithms like Random Forest. Some statisticians have adopted methods from machine learning, leading to a combined field that they call statistical learning. Analytical and computational techniques derived from deep-rooted physics of disordered systems can be extended to large-scale problems, including machine learning, e.g., to analyse the weight space of deep neural networks. Statistical physics is thus finding applications in the area of medical diagnostics. A core objective of a learner is to generalise from its experience. Generalisation in this context is the ability of a learning machine to perform accurately on new, unseen examples/tasks after having experienced a learning data set. The training examples come from some generally unknown probability distribution (considered representative of the space of occurrences) and the learner has to build a general model about this space that enables it to produce sufficiently accurate predictions in new cases. The computational analysis of machine learning algorithms and their performance is a branch of theoretical computer science known as computational learning theory via the probably approximately correct learning model. Because training sets are finite and the future is uncertain, learning theory usually does not yield guarantees of the performance of algorithms. Instead, probabilistic bounds on the performance are quite common. The bias-variance decomposition is one way to quantify generalisation error. For the best performance in the context of generalisation, the complexity of the hypothesis should match the complexity of the function underlying the data. If the hypothesis is less complex than the function, then the model has under fitted the data. If the complexity of the model is increased in response, then the training error decreases. But if the hypothesis is too complex, then the model is subject to overfitting and generalisation will be poorer. In addition to performance bounds, learning theorists study the time complexity and feasibility of learning. In computational learning theory, a computation is considered feasible if it can be done in polynomial time. There are two kinds of time complexity results: Positive results show that a certain class of functions can be learned in polynomial time. Negative results show that certain classes cannot be learned in polynomial time. Machine learning approaches are traditionally divided into three broad categories, which correspond to learning paradigms, depending on the nature of the \"signal\" or \"feedback\" available to the learning system: Supervised learning: The computer is presented with example inputs and their desired outputs, given by a \"teacher\", and the goal is to learn a general rule that maps inputs to outputs. Unsupervised learning: No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning). Reinforcement learning: A computer program interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle or playing a game against an opponent). As it navigates its problem space, the program is provided feedback that's analogous to rewards, which it tries to maximise. Although each algorithm has advantages and limitations, no single algorithm works for all problems. Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. The data, known as training data, consists of a set of training examples. Each training example has one or more inputs and the desired output, also known as a supervisory signal. In the mathematical model, each training example is represented by an array or vector, sometimes called a feature vector, and the training data is represented by a matrix. Through iterative optimisation of an objective function, supervised learning algorithms learn a function that can be used to predict the output associated with new inputs. An optimal function allows the algorithm to correctly determine the output for inputs that were not a part of the training data. An algorithm that improves the accuracy of its outputs or predictions over time is said to have learned to perform that task. Types of supervised-learning algorithms include active learning, classification and regression. Classification algorithms are used when the outputs are restricted to a limited set of values, while regression algorithms are used when the outputs can take any numerical value within a range. For example, in a classification algorithm that filters emails, the input is an incoming email, and the output is the folder in which to file the email. In contrast, regression is used for tasks such as predicting a person's height based on factors like age and genetics or forecasting future temperatures based on historical data. Similarity learning is an area of supervised machine learning closely related to regression and classification, but the goal is to learn from examples using a similarity function that measures how similar or related two objects are. It has applications in ranking, recommendation systems, visual identity tracking, face verification, and speaker verification. Unsupervised learning algorithms find structures in data that has not been labelled, classified or categorised. Instead of responding to feedback, unsupervised learning algorithms identify commonalities in the data and react based on the presence or absence of such commonalities in each new piece of data. Central applications of unsupervised machine learning include clustering, dimensionality reduction, and density estimation. Cluster analysis is the assignment of a set of observations into subsets (called clusters) so that observations within the same cluster are similar according to one or more predesignated criteria, while observations drawn from different clusters are dissimilar. Different clustering techniques make different assumptions on the structure of the data, often defined by some similarity metric and evaluated, for example, by internal compactness, or the similarity between members of the same cluster, and separation, the difference between clusters. Other methods are based on estimated density and graph connectivity. A special type of unsupervised learning called, self-supervised learning involves training a model by generating the supervisory signal from the data itself. Semi-supervised learning falls between unsupervised learning (without any labelled training data) and supervised learning (with completely labelled training data). Some of the training examples are missing training labels, yet many machine-learning researchers have found that unlabelled data, when used in conjunction with a small amount of labelled data, can produce a considerable improvement in learning accuracy. In weakly supervised learning, the training labels are noisy, limited, or imprecise; however, these labels are often cheaper to obtain, resulting in larger effective training sets. Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximise some notion of cumulative reward. Due to its generality, the field is studied in many other disciplines, such as game theory, control theory, operations research, information theory, simulation-based optimisation, multi-agent systems, swarm intelligence, statistics and genetic algorithms. In reinforcement learning, the environment is typically represented as a Markov decision process (MDP). Many reinforcement learning algorithms use dynamic programming techniques. Reinforcement learning algorithms do not assume knowledge of an exact mathematical model of the MDP and are used when exact models are infeasible. Reinforcement learning algorithms are used in autonomous vehicles or in learning to play a game against a human opponent. Dimensionality reduction is a process of reducing the number of random variables under consideration by obtaining a set of principal variables. In other words, it is a process of reducing the dimension of the feature set, also called the \"number of features\". Most of the dimensionality reduction techniques can be considered as either feature elimination or extraction. One of the popular methods of dimensionality reduction is principal component analysis (PCA). PCA involves changing higher-dimensional data (e.g., 3D) to a smaller space (e.g., 2D). The manifold hypothesis proposes that high-dimensional data sets lie along low-dimensional manifolds, and many dimensionality reduction techniques make this assumption, leading to the area of manifold learning and manifold regularisation. Other approaches have been developed which do not fit neatly into this three-fold categorisation, and sometimes more than one is used by the same machine learning system. For example, topic modelling, meta-learning. Self-learning, as a machine learning paradigm was introduced in 1982 along with a neural network capable of self-learning, named crossbar adaptive array (CAA). It gives a solution to the problem learning without any external reward, by introducing emotion as an internal reward. Emotion is used as state evaluation of a self-learning agent. The CAA self-learning algorithm computes, in a crossbar fashion, both decisions about actions and emotions (feelings) about consequence situations. The system is driven by the interaction between cognition and emotion. The self-learning algorithm updates a memory matrix W =||w(a,s)|| such that in each iteration executes the following machine learning routine: in situation s perform action a receive a consequence situation s' compute emotion of being in the consequence situation v(s') update crossbar memory w'(a,s) = w(a,s) + v(s') It is a system with only one input, situation, and only one output, action (or behaviour) a. There is neither a separate reinforcement input nor an advice input from the environment. The backpropagated value (secondary reinforcement) is the emotion toward the consequence situation. The CAA exists in two environments, one is the behavioural environment where it behaves, and the other is the genetic environment, wherefrom it initially and only once receives initial emotions about situations to be encountered in the behavioural environment. After receiving the genome (species) vector from the genetic environment, the CAA learns a goal-seeking behaviour, in an environment that contains both desirable and undesirable situations. Several learning algorithms aim at discovering better representations of the inputs provided during training. Classic examples include principal component analysis and cluster analysis. Feature learning algorithms, also called representation learning algorithms, often attempt to preserve the information in their input but also transform it in a way that makes it useful, often as a pre-processing step before performing classification or predictions. This technique allows reconstruction of the inputs coming from the unknown data-generating distribution, while not being necessarily faithful to configurations that are implausible under that distribution. This replaces manual feature engineering, and allows a machine to both learn the features and use them to perform a specific task. Feature learning can be either supervised or unsupervised. In supervised feature learning, features are learned using labelled input data. Examples include artificial neural networks, multilayer perceptrons, and supervised dictionary learning. In unsupervised feature learning, features are learned with unlabelled input data. Examples include dictionary learning, independent component analysis, autoencoders, matrix factorisation and various forms of clustering. Manifold learning algorithms attempt to do so under the constraint that the learned representation is low-dimensional. Sparse coding algorithms attempt to do so under the constraint that the learned representation is sparse, meaning that the mathematical model has many zeros. Multilinear subspace learning algorithms aim to learn low-dimensional representations directly from tensor representations for multidimensional data, without reshaping them into higher-dimensional vectors. Deep learning algorithms discover multiple levels of representation, or a hierarchy of features, with higher-level, more abstract features defined in terms of (or generating) lower-level features. It has been argued that an intelligent machine is one that learns a representation that disentangles the underlying factors of variation that explain the observed data.
            """,
            metadata={"source": "neural_networks.txt", "title": "Neural Networks"}
        )
    ]

    print(f"✅ Created {len(docs)} dummy documents")
    for i, doc in enumerate(docs):
        print(f"   Doc {i + 1}: {doc.metadata['title']} ({len(doc.page_content)} chars)")

    return docs


def test_ragas_standard_pipeline():
    """Test RAGAS using its standard pipeline to see what works."""

    print("🧪 Testing RAGAS Standard Pipeline")
    print("=" * 50)

    try:
        # Import RAGAS components
        from ragas.testset.graph import KnowledgeGraph, Node, NodeType
        from ragas.testset.transforms import default_transforms, apply_transforms
        from ragas.testset import TestsetGenerator
        from ragas.testset.synthesizers import default_query_distribution
        from ragas.testset.persona import Persona
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        print("✅ RAGAS imports successful")

    except ImportError as e:
        print(f"❌ RAGAS import failed: {e}")
        print("💡 Try: pip install ragas langchain-openai")
        return

    # Create dummy documents
    docs = create_dummy_documents()

    # Setup LLMs
    print("\n🤖 Setting up LLMs...")
    try:
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7))
        generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-ada-002"))
        print("✅ LLMs initialized successfully")
    except Exception as e:
        print(f"❌ LLM setup failed: {e}")
        print("💡 Check your OpenAI API key")
        return

    # Step 1: Create Knowledge Graph using RAGAS standard approach
    print("\n📊 Creating Knowledge Graph...")
    kg = KnowledgeGraph()

    # Add documents as nodes (RAGAS standard way)
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata
                }
            )
        )

    print(f"✅ Added {len(kg.nodes)} document nodes")

    # Step 2: Apply RAGAS default transforms
    print("\n🔧 Applying RAGAS transforms...")
    try:
        transforms = default_transforms(
            documents=docs,
            llm=generator_llm,
            embedding_model=generator_embeddings
        )

        print(f"   Transforms to apply: {len(transforms)}")
        for i, transform in enumerate(transforms):
            print(f"     {i + 1}. {transform.__class__.__name__}")

        apply_transforms(kg, transforms)
        print(f"✅ Transforms applied successfully")

    except Exception as e:
        print(f"❌ Transform application failed: {e}")
        print(f"   Error details: {str(e)}")
        return

    # Step 3: Inspect the resulting knowledge graph
    print(f"\n🔍 Inspecting Knowledge Graph...")
    print(f"   Total nodes: {len(kg.nodes)}")
    print(f"   Total relationships: {len(kg.relationships)}")

    # Show node types
    node_types = {}
    for node in kg.nodes:
        node_types[node.type] = node_types.get(node.type, 0) + 1
    print(f"   Node types: {node_types}")

    # Show relationship types
    if kg.relationships:
        rel_types = {}
        for rel in kg.relationships:
            rel_types[rel.type] = rel_types.get(rel.type, 0) + 1
        print(f"   Relationship types: {rel_types}")

        # Show sample relationships
        print(f"   Sample relationships:")
        for i, rel in enumerate(kg.relationships[:3]):
            print(f"     {i + 1}. {rel.type}: {rel.source} -> {rel.target}")
    else:
        print("   ⚠️  No relationships found!")

    # Show sample node properties
    if kg.nodes:
        print(f"   Sample node properties:")
        sample_node = kg.nodes[0]
        for key in list(sample_node.properties.keys())[:5]:
            value = sample_node.properties[key]
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            print(f"     {key}: {value}")

    # Step 4: Create simple personas
    print(f"\n👥 Creating personas...")
    personas = [
        Persona(
            name="AI Researcher",
            role_description="Researcher studying artificial intelligence and machine learning techniques. Asks detailed questions about algorithms and methodologies."
        ),
        Persona(
            name="Student",
            role_description="Computer science student learning about AI concepts. Needs clear explanations and examples."
        ),
        Persona(
            name="Engineer",
            role_description="Software engineer implementing AI solutions. Focused on practical applications and implementation details."
        )
    ]
    print(f"✅ Created {len(personas)} personas")

    # Step 5: Try question generation
    print(f"\n🎯 Generating questions...")
    try:
        # Create generator
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            knowledge_graph=kg,
            persona_list=personas
        )

        # Use default query distribution
        query_distribution = default_query_distribution(generator_llm)
        print(f"   Query distribution: {len(query_distribution)} synthesizer types")

        # Generate small test set
        testset = generator.generate(
            testset_size=5,
            query_distribution=query_distribution,
            num_personas=len(personas)
        )

        print(f"✅ Successfully generated {len(testset)} questions!")

        # Show generated questions
        df = testset.to_pandas()
        print(f"\n📝 Generated Questions:")
        for i, row in df.iterrows():
            print(f"   {i + 1}. {row['user_input']}")
            if hasattr(row, 'synthesizer_name'):
                print(f"      Type: {row.get('synthesizer_name', 'unknown')}")
            print()

        return True

    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error details: {str(e)}")

        # Try to get more debug info
        import traceback
        print(f"\n🔍 Full traceback:")
        traceback.print_exc()

        return False


def save_knowledge_graph_structure(kg, filename="ragas_kg_structure.json"):
    """Save the knowledge graph structure for inspection."""
    try:
        kg_data = {
            "nodes": [
                {
                    "id": node.id,
                    "type": str(node.type),
                    "properties_keys": list(node.properties.keys()),
                    "sample_properties": {
                        k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                        for k, v in list(node.properties.items())[:3]
                    }
                }
                for node in kg.nodes
            ],
            "relationships": [
                {
                    "source": rel.source,
                    "target": rel.target,
                    "type": rel.type,
                    "properties": rel.properties
                }
                for rel in kg.relationships
            ],
            "summary": {
                "total_nodes": len(kg.nodes),
                "total_relationships": len(kg.relationships),
                "node_types": {str(node.type): sum(1 for n in kg.nodes if n.type == node.type) for node in kg.nodes},
                "relationship_types": {rel.type: sum(1 for r in kg.relationships if r.type == rel.type) for rel in
                                       kg.relationships}
            }
        }

        with open(filename, 'w') as f:
            json.dump(kg_data, f, indent=2)

        print(f"💾 Knowledge graph structure saved to {filename}")

    except Exception as e:
        print(f"⚠️  Could not save KG structure: {e}")


def main():
    """Main test function."""
    print("🧪 RAGAS Minimal Test Script")
    print("Testing what RAGAS needs for successful question generation")
    print("=" * 70)

    # Check API key
    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("❌ Please set your OpenAI API key in the script!")
        print("   os.environ['OPENAI_API_KEY'] = 'your-actual-key-here'")
        return

    success = test_ragas_standard_pipeline()

    if success:
        print("\n✅ SUCCESS! RAGAS question generation worked!")
        print("🔍 Check the console output above to see:")
        print("   • What node types RAGAS created")
        print("   • What relationship types RAGAS uses")
        print("   • What properties are expected")
        print("   • What the successful knowledge graph looks like")
    else:
        print("\n❌ FAILED! But we learned something about what's missing.")
        print("🔍 Check the error messages above to understand what RAGAS expects.")


if __name__ == "__main__":
    main()