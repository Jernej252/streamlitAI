# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!

# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")

    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!

    my_documents = [
        # 1. Sports Technology
        """Technology is rewriting the playbook across every arena. Wearable sensors embedded in compression shirts track heart‑rate variability and core temperature in real time, allowing coaches to pull athletes before fatigue escalates into injury. GPS and inertial units log sprint counts, high‑speed distance, even jump load, feeding dashboards that fine‑tune training loads per individual. Video‑assisted refereeing (VAR) and Hawk‑Eye line‑calling blend high‑frame‑rate cameras with computer vision algorithms, turning controversial calls into evidence‑based rulings within seconds. In equipment, 3‑D‑printed lattice midsoles tailor running‑shoe cushioning to each athlete’s force plate data, while carbon‑fiber track spikes leverage energy‑return plates that shave milliseconds off personal bests. Data analytics platforms ingest millions of plays to uncover hidden patterns—think NBA line‑up efficiencies or MLB pitch sequencing—informing strategy that once relied on gut instinct. Fan experience evolves too: augmented‑reality apps overlay live stats onto smartphones, and volumetric video lets viewers spin replays in 360 degrees. As processors shrink and AI models scale, the boundary between athlete and algorithm keeps receding.""",

        # 2. Olympic Games History
        """The Olympic Games trace their lineage to 776 BCE in Olympia, Greece, where city‑states paused wars to watch heralded athletes sprint the stadion. After nearly twelve centuries, the Roman emperor Theodosius I banned pagan festivals in 393 CE, extinguishing the flame. Modern revivalist Pierre de Coubertin resurrected the ideal in 1896, staging the first contemporary Games in Athens with 241 athletes from 14 nations. Milestones quickly followed: women debuted in 1900 Paris, Paavo Nurmi’s distance dominance in the 1920s symbolized international heroism, Jesse Owens shattered Nazi propaganda in Berlin 1936, while the 1960 Rome Games introduced worldwide live television. The Olympics have weathered political boycotts (Moscow 1980, Los Angeles 1984), tragic violence (Munich 1972), and record‑breaking spectacles like Beijing 2008’s choreographed opening ceremony. Today over 11 000 athletes compete across summer and winter editions, carrying a torch that physically relays through thousands of hands and culturally conveys the pursuit of excellence, friendship, and respect that Coubertin envisioned more than a century ago.""",

        # 3. Mental Skills in Elite Sport
        """Elite sport is as much a neurological contest as a physical one. Visualization primes neural pathways: an athlete rehearses a perfect free‑throw or vault run‑up in vivid, multisensory detail, firing the same motor neurons that activate during actual execution. Goal setting converts dreams into benchmarks—use the SMART framework so objectives are specific, measurable, attainable, relevant, and time‑bound, then break them into daily process goals that sustain focus amid adversity. Self‑talk scripts the internal narrative; swapping “don’t miss” for “sink the shot” frames outcomes positively, while cue words like “explode” or “steady” trigger automatic technique cues. Mindfulness meditation trains awareness of breath and body sensations, dampening the amygdala’s stress response and restoring present‑moment clarity. Pre‑performance routines—consistent sequences of stretches, breaths, or mantras—signal the brain to enter flow state. Finally, resilience grows through controlled exposure to pressure: practice clutch scenarios at the end of training, so competition feels familiar. Mental reps, like physical ones, compound into unbeatable confidence.""",

        # 4. Nutrition for Athletes
        """Peak athletic performance begins in the kitchen long before it shows on the scoreboard. Carbohydrates are the primary fuel; choose complex sources—oats, brown rice, quinoa—so glycogen stores top off gradually, sustaining energy through practice. Protein repairs muscle micro‑tears; target 1.6–2.0 g per kilogram body weight daily from lean meats, legumes, or dairy, spaced every 3–4 hours to maintain positive nitrogen balance. Healthy fats, roughly 25–30 % of total calories, modulate hormones and dampen inflammation; emphasize omega‑3‑rich salmon, walnuts, and flax. Hydration is non‑negotiable: begin each session already euhydrated, sip 150–250 ml every 15 minutes, and replace 1.5 times any fluid lost post‑workout. Timing matters: a 3:1 carb‑to‑protein snack within 45 minutes of training accelerates recovery, while nitrate‑dense beet juice consumed 2 hours prior boosts endurance by enhancing oxygen economy. Micronutrients round out the picture—iron for oxygen transport, vitamin D for bone health, and antioxidants to quench exercise‑induced free radicals—ensuring the body’s engine fires efficiently when the whistle blows.""",

        # 5. Basketball Fundamentals
        """Basketball success rests on flawless execution of a few timeless fundamentals. Dribbling must stay controlled and low—use your fingertips, not the palm, to keep the ball below the waist and protect it with the off‑hand as you change speeds. Passing is the game’s heartbeat: snap chest passes to a teammate’s torso, bounce passes two‑thirds of the way, and master the overhead outlet to ignite transition. Shooting begins in the feet; align toes, bend knees, and generate a smooth upward energy transfer that ends with a relaxed, high‑arc follow‑through. From the triple‑threat stance—ball at hip, knees bent—players can shoot, drive, or pass in a single motion, forcing defenders to hesitate. Defensive fundamentals mirror the offense: stay low in a staggered stance, keep eyes on the opponent’s torso, and slide (do not cross) the feet to cut off driving lanes. Rebounding crowns every play; locate the opponent, box out with contact, then explode toward the rim two hands high. Repetition of these basics builds instinct, allowing creativity to flourish when the clock is ticking."""
    ]

    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )

    return collection

def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 1.5:  # 1.5 is similarity threshold - adjust as needed
        return "I don't have information about that topic in my documents."
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=150
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    return answer

# MAIN APP STARTS HERE - This is where we build the user interface


# --- CUSTOMIZED APPEARANCE ---

st.title("🏆 Sports Knowledge Powerhouse")

st.markdown("""
### 🏅 Welcome to **Your Ultimate Sports Q&A System!**
*Ask anything about technology in sports, Olympic history, mental skills, nutrition, or basketball fundamentals.*
""")

st.info("🤓 Get expert answers from my hand-crafted sports knowledge base!", icon="🏆")

collection = setup_documents()

question = st.text_input("What would you like to know about sports, training, or competitions?")

if st.button("🏅 Get My Sports Answer!", type="primary"):
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    if question:
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        with st.spinner("⏳ Crunching the stats and searching the playbook..."):
            answer = get_answer(collection, question)
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        st.markdown("**Answer:**")
        st.success(f"{answer}")
    else:
        st.warning("⚠️ Please enter a question to get started!")

with st.expander("About this Sports Q&A System 🏆"):
    st.write("""
    I created this system with knowledge about:
    - Technology in sports (wearables, analytics, equipment)
    - Olympic Games history and milestones
    - Mental skills and psychology in elite sport
    - Nutrition for athletes and performance
    - Basketball fundamentals and techniques
    
    **Try asking about sports tech, Olympic facts, mental training, nutrition, or basketball skills!**
    """)

# TO RUN: Save as app.py, then type: streamlit run app.py
##
# STREAMLIT BUILDING BLOCKS SUMMARY:
# =================================
# 1. st.title(text) - Creates the main heading of your app
# 2. st.write(text) - Displays text, data, or markdown content
# 3. st.text_input(label, placeholder="hint") - Creates a text box for user input
# 4. st.button(text, type="primary") - Creates a clickable button
# 5. st.spinner(text) - Shows a spinning animation with custom text
# 6. st.expander(title) - Creates a collapsible section
# HOW THE APP FLOW WORKS:
# 1. User opens browser → Streamlit loads the app
# 2. setup_documents() runs → Creates document database
# 3. st.title() and st.write() → Display app header
# 4. st.text_input() → Shows input box for questions
# 5. st.button() → Shows the "Get Answer" button
# 6. User types question and clicks button:
#    - if statement triggers
#    - st.spinner() shows loading animation
#    - get_answer() function runs in background
#    - st.write() displays the result
# 7. st.expander() → Shows help section at bottom
# WHAT HAPPENS WHEN USER INTERACTS:
# - Type in text box → question variable updates automatically
# - Click button → if st.button() becomes True
# - Spinner shows → get_answer() function runs
# - Answer appears → st.write() displays the result
# - Click expander → help section shows/hides
# This creates a simple but complete web application!
