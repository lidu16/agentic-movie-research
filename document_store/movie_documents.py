"""
Movie Documents - Detailed information about movies for vector RAG.
"""

MOVIE_DOCUMENTS = [
    {
        "id": "matrix",
        "title": "The Matrix",
        "year": 1999,
        "runtime": "136 minutes",
        "genre": "Science Fiction, Action",
        "directors": "Lana Wachowski, Lilly Wachowski",
        "description": """
The Matrix is a groundbreaking 1999 science fiction action film that explores the nature of reality 
and human existence. The story follows Thomas Anderson, a computer programmer who leads a secret 
life as a hacker known as "Neo". He discovers that the world he lives in is actually a simulated 
reality called "The Matrix", created by sentient machines to subdue and control the human population 
while using their bodies as an energy source.

Neo is contacted by the legendary hacker Morpheus, who believes Neo is "The One" - a prophesied 
figure destined to end the war between humans and machines. Along with the skilled warrior Trinity, 
Morpheus frees Neo from the Matrix and shows him the true devastated state of the real world.

The film is renowned for its innovative visual effects, particularly the "bullet time" technique, 
philosophical themes drawing from Plato's Allegory of the Cave, Buddhist and Christian symbolism, 
and cyberpunk aesthetics. It explores questions about free will, determinism, and the nature of 
consciousness.

Key themes: Reality vs illusion, chosen one prophecy, human-machine conflict, red pill/blue pill choice,
martial arts combat, computer hacking, dystopian future, philosophical awakening.
        """,
        "cast": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving",
        "box_office": "$466.6 million worldwide",
        "awards": "4 Academy Awards (Best Film Editing, Sound, Sound Effects Editing, Visual Effects)"
    },
    {
        "id": "forrest_gump",
        "title": "Forrest Gump",
        "year": 1994,
        "runtime": "142 minutes",
        "genre": "Drama, Romance, Comedy",
        "directors": "Robert Zemeckis",
        "description": """
Forrest Gump is a 1994 American epic comedy-drama film that chronicles the life of Forrest Gump, 
a kind-hearted man from Alabama with a low IQ but good intentions, who witnesses and unwittingly 
influences several major historical events in the 20th century United States.

Despite his intellectual limitations, Forrest achieves remarkable success in various fields. He 
becomes a college football star, serves with honor in the Vietnam War where he saves several 
fellow soldiers including his commanding officer Lieutenant Dan, becomes a world-class ping-pong 
player, starts a successful shrimp fishing business with his friend Bubba's family, and becomes 
a millionaire.

Throughout his extraordinary journey, Forrest remains devoted to his childhood love, Jenny Curran, 
whose troubled life takes her on a very different path through the counterculture of the 1960s and 
1970s. The film uses groundbreaking visual effects to place Forrest in historical footage with 
figures like Presidents Kennedy, Johnson, and Nixon.

The movie's famous quote "Life is like a box of chocolates, you never know what you're gonna get" 
became a cultural phenomenon. It explores themes of destiny vs free will, the American Dream, 
unconditional love, and how an ordinary person can impact extraordinary events.

Key themes: American history, Vietnam War, unconditional love, destiny, innocence, perseverance,
the 1960s-80s era, disability representation, friendship, running across America.
        """,
        "cast": "Tom Hanks, Robin Wright, Gary Sinise, Sally Field",
        "box_office": "$678.2 million worldwide",
        "awards": "6 Academy Awards (Best Picture, Best Director, Best Actor, Best Adapted Screenplay, Best Film Editing, Best Visual Effects)"
    },
    {
        "id": "inception",
        "title": "Inception",
        "year": 2010,
        "runtime": "148 minutes",
        "genre": "Science Fiction, Action, Thriller",
        "directors": "Christopher Nolan",
        "description": """
Inception is a 2010 science fiction heist thriller written and directed by Christopher Nolan. 
The film explores the concept of shared dreaming and the architecture of the subconscious mind.

Dom Cobb is a skilled thief who specializes in "extraction" - the art of entering people's dreams 
to steal valuable secrets from their subconscious. His abilities have made him a valuable player 
in corporate espionage but have also cost him everything he loves. Cobb is offered a chance at 
redemption when he's given an impossible task: instead of stealing an idea, he must plant one - 
a process called "inception."

To accomplish this, Cobb assembles a team of specialists: Arthur (his research and planning expert), 
Ariadne (a young architect who designs the dream worlds), Eames (a forger who can impersonate others 
in dreams), Yusuf (a chemist who creates powerful sedatives), and Saito (the businessman who hired 
them). Their target is Robert Fischer, heir to a massive business empire.

The film features multiple nested dream levels, each with different time dilation - five minutes 
in reality becomes an hour in the first dream level, and time expands exponentially in deeper 
levels. The iconic spinning top "totem" is used by Cobb to distinguish reality from dreams.

Key themes: Dreams within dreams, subconscious mind, guilt and memory, reality perception, 
heist planning, time dilation, limbo, totems, grief processing, corporate espionage.
        """,
        "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page, Tom Hardy, Ken Watanabe",
        "box_office": "$836.8 million worldwide",
        "awards": "4 Academy Awards (Best Cinematography, Sound Mixing, Sound Editing, Visual Effects)"
    }
]


def get_all_documents() -> list[dict]:
    """Get all movie documents."""
    return MOVIE_DOCUMENTS


def get_document_texts() -> list[tuple[str, str]]:
    """Get (id, text) pairs for embedding."""
    docs = []
    for movie in MOVIE_DOCUMENTS:
        # Combine all text fields for embedding
        text = f"""
Title: {movie['title']} ({movie['year']})
Genre: {movie['genre']}
Runtime: {movie['runtime']}
Directors: {movie['directors']}
Cast: {movie['cast']}

{movie['description']}

Box Office: {movie['box_office']}
Awards: {movie['awards']}
        """.strip()
        docs.append((movie['id'], text))
    return docs
