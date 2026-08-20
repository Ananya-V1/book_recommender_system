import os
import sqlite3
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import gradio as gr
import time

load_dotenv()
client = OpenAI()

DATA_FILE = "books.csv"
EMBED_FILE = "books_with_embeddings.pkl"
BATCH_SIZE = 50
DB_FILE = "chatbot.db"

# -----------------------------
# Database setup
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS books_read (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            authors TEXT,
            categories TEXT,
            average_rating TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS books_want (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            authors TEXT,
            categories TEXT,
            average_rating TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS friendships (
            user_id INTEGER NOT NULL,
            friend_id INTEGER NOT NULL,
            PRIMARY KEY (user_id, friend_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (friend_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS friend_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_user_id INTEGER NOT NULL,
            to_user_id INTEGER NOT NULL,
            UNIQUE (from_user_id, to_user_id),
            FOREIGN KEY (from_user_id) REFERENCES users(id),
            FOREIGN KEY (to_user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_by INTEGER NOT NULL,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS group_members (
            group_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            PRIMARY KEY (group_id, user_id),
            FOREIGN KEY (group_id) REFERENCES groups(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Auth
# -----------------------------
def register(username, password):
    if not username.strip() or not password.strip():
        return None, "❌ Username and password required"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                  (username.strip(), generate_password_hash(password, method="pbkdf2:sha256")))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return {"id": user_id, "username": username.strip()}, f"✅ Account created! Welcome, {username.strip()}!"
    except sqlite3.IntegrityError:
        conn.close()
        return None, "❌ Username already taken"

def login(username, password):
    if not username.strip() or not password.strip():
        return None, "❌ Username and password required"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username.strip(),))
    row = c.fetchone()
    conn.close()
    if row and check_password_hash(row[1], password):
        return {"id": row[0], "username": username.strip()}, f"✅ Welcome back, {username.strip()}!"
    return None, "❌ Invalid username or password"

# -----------------------------
# Book helpers
# -----------------------------
def get_user_books(user_id, table):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        f"SELECT title, authors, categories, average_rating FROM {table} WHERE user_id = ?",
        conn, params=(user_id,)
    )
    conn.close()
    return df

def add_book(user_id, title, authors, categories, rating, table):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"SELECT id FROM {table} WHERE user_id = ? AND title = ?", (user_id, title))
    if not c.fetchone():
        c.execute(
            f"INSERT INTO {table} (user_id, title, authors, categories, average_rating) VALUES (?, ?, ?, ?, ?)",
            (user_id, title, authors, categories, rating)
        )
        conn.commit()
    conn.close()

def remove_book(user_id, title, table):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"DELETE FROM {table} WHERE user_id = ? AND title = ?", (user_id, title))
    conn.commit()
    conn.close()

def get_book_titles(user_id, table):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"SELECT title FROM {table} WHERE user_id = ?", (user_id,))
    titles = [row[0] for row in c.fetchall()]
    conn.close()
    return titles

def get_book_status(user_id, title):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM books_read WHERE user_id = ? AND title = ?", (user_id, title))
    if c.fetchone():
        conn.close()
        return "✅ Read"
    c.execute("SELECT id FROM books_want WHERE user_id = ? AND title = ?", (user_id, title))
    if c.fetchone():
        conn.close()
        return "⭐ Want to Read"
    conn.close()
    return ""

# -----------------------------
# Friends helpers
# -----------------------------
def send_friend_request(user_id, friend_username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (friend_username.strip(),))
    row = c.fetchone()
    if not row:
        conn.close()
        return "❌ User not found"
    friend_id = row[0]
    if friend_id == user_id:
        conn.close()
        return "❌ You can't add yourself"
    c.execute("SELECT 1 FROM friendships WHERE user_id = ? AND friend_id = ?", (user_id, friend_id))
    if c.fetchone():
        conn.close()
        return f"❌ Already friends with {friend_username.strip()}"
    try:
        c.execute("INSERT INTO friend_requests (from_user_id, to_user_id) VALUES (?, ?)", (user_id, friend_id))
        conn.commit()
        conn.close()
        return f"✅ Friend request sent to {friend_username.strip()}!"
    except sqlite3.IntegrityError:
        conn.close()
        return "❌ Friend request already sent"

def get_pending_requests(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT fr.id, u.username FROM friend_requests fr
        JOIN users u ON u.id = fr.from_user_id
        WHERE fr.to_user_id = ?
    """, (user_id,))
    requests = [(row[0], row[1]) for row in c.fetchall()]
    conn.close()
    return requests

def accept_request(request_id, user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT from_user_id FROM friend_requests WHERE id = ? AND to_user_id = ?", (request_id, user_id))
    row = c.fetchone()
    if not row:
        conn.close()
        return "❌ Request not found"
    from_id = row[0]
    c.execute("INSERT OR IGNORE INTO friendships (user_id, friend_id) VALUES (?, ?)", (user_id, from_id))
    c.execute("INSERT OR IGNORE INTO friendships (user_id, friend_id) VALUES (?, ?)", (from_id, user_id))
    c.execute("DELETE FROM friend_requests WHERE id = ?", (request_id,))
    conn.commit()
    conn.close()
    return "✅ Friend request accepted!"

def decline_request(request_id, user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM friend_requests WHERE id = ? AND to_user_id = ?", (request_id, user_id))
    conn.commit()
    conn.close()
    return "❌ Request declined"

def get_friends(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT u.username FROM users u
        JOIN friendships f ON f.friend_id = u.id
        WHERE f.user_id = ?
    """, (user_id,))
    friends = [row[0] for row in c.fetchall()]
    conn.close()
    return friends

def get_friend_books(user_id, friend_username, table):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (friend_username,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None, "❌ User not found"
    friend_id = row[0]
    c.execute("SELECT 1 FROM friendships WHERE user_id = ? AND friend_id = ?", (user_id, friend_id))
    if not c.fetchone():
        conn.close()
        return None, "❌ You're not friends with this user"
    df = pd.read_sql_query(
        f"SELECT title, authors, categories, average_rating FROM {table} WHERE user_id = ?",
        conn, params=(friend_id,)
    )
    conn.close()
    return df, None

# -----------------------------
# Load embeddings
# -----------------------------
def load_data():
    if os.path.exists(EMBED_FILE):
        print("✅ Loading cached embeddings...")
        df = pd.read_pickle(EMBED_FILE)
    else:
        df = pd.read_csv(DATA_FILE)
        df = df[["title", "authors", "categories", "description", "average_rating"]]
        df = df.dropna(subset=["description"])
        df = df.fillna("").astype(str)
        df["text"] = (
            "Title: " + df["title"] +
            "\nAuthor: " + df["authors"] +
            "\nGenre: " + df["categories"] +
            "\nDescription: " + df["description"]
        )
        df["text"] = df["text"].apply(lambda x: x[:2000])
        df["embedding"] = None

    start_idx = df.index[df["embedding"].isnull()].min() if df["embedding"].isnull().any() else None
    if start_idx is None:
        print("💾 All embeddings already exist.")
        return df

    total = len(df)
    for i in range(start_idx, total, BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]
        print(f"Processing books {i+1} to {i+len(batch)}...")
        batch_texts = batch["text"].tolist()
        response = client.embeddings.create(model="text-embedding-3-small", input=batch_texts)
        clean_embeddings = []
        for emb in response.data:
            clean_emb = []
            for x in emb.embedding:
                try:
                    f = float(x)
                    if np.isnan(f) or np.isinf(f):
                        f = 0.0
                except:
                    f = 0.0
                clean_emb.append(f)
            clean_embeddings.append(clean_emb)
        if len(clean_embeddings) != len(batch):
            raise ValueError(f"Batch size mismatch: {len(batch)} vs {len(clean_embeddings)}")
        for idx, emb in zip(batch.index, clean_embeddings):
            df.at[idx, "embedding"] = emb
        df.to_pickle(EMBED_FILE)
        print(f"💾 Saved progress after batch ending at book {i+len(batch)}")
        time.sleep(1)

    print("✅ All embeddings complete!")
    return df

df = load_data()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, user, last_results):
    if not user:
        return None, last_results, "❌ Please log in first"
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    query_emb = [float(x) if not (np.isnan(x) or np.isinf(x)) else 0.0
                 for x in response.data[0].embedding]
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, query_emb))
    results = df.sort_values("similarity", ascending=False).head(5).copy()
    results["status"] = results["title"].apply(lambda t: get_book_status(user["id"], t))
    results = results[["status", "title", "authors", "categories", "average_rating", "similarity"]]
    return results, results, ""

def add_selected_book(index, list_type, user, last_results):
    if not user:
        return "❌ Please log in first"
    if last_results is None:
        return "❌ No search results yet"
    try:
        row = last_results.iloc[int(index)]
    except:
        return "❌ Invalid index"
    table = "books_read" if list_type == "read" else "books_want"
    label = "READ" if list_type == "read" else "WANT-TO-READ"
    add_book(user["id"], row["title"], row["authors"], row["categories"], row["average_rating"], table)
    return f"✅ Added '{row['title']}' to {label} list"

def track_add_read(title, authors, categories, rating, user):
    if not user:
        return "❌ Please log in first"
    add_book(user["id"], title, authors, categories, rating, "books_read")
    return f"✅ Added '{title}' to READ list"

def track_add_want(title, authors, categories, rating, user):
    if not user:
        return "❌ Please log in first"
    add_book(user["id"], title, authors, categories, rating, "books_want")
    return f"⭐ Added '{title}' to WANT-TO-READ list"

# -----------------------------
# Settings
# -----------------------------
def change_password(user_id, current_password, new_password):
    if not current_password or not new_password:
        return "❌ All fields required"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    if not row or not check_password_hash(row[0], current_password):
        conn.close()
        return "❌ Current password is incorrect"
    c.execute("UPDATE users SET password_hash = ? WHERE id = ?",
              (generate_password_hash(new_password, method="pbkdf2:sha256"), user_id))
    conn.commit()
    conn.close()
    return "✅ Password changed successfully"

# -----------------------------
# Leaderboard
# -----------------------------
def get_global_leaderboard():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT u.username, COUNT(b.id) as books_read
        FROM users u
        LEFT JOIN books_read b ON b.user_id = u.id
        GROUP BY u.id
        ORDER BY books_read DESC
    """, conn)
    conn.close()
    df.index = df.index + 1
    df.index.name = "rank"
    return df.reset_index()

# -----------------------------
# Groups
# -----------------------------
def create_group(user_id, group_name):
    if not group_name.strip():
        return "❌ Group name required"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO groups (name, created_by) VALUES (?, ?)", (group_name.strip(), user_id))
        group_id = c.lastrowid
        c.execute("INSERT INTO group_members (group_id, user_id) VALUES (?, ?)", (group_id, user_id))
        conn.commit()
        conn.close()
        return f"✅ Group '{group_name.strip()}' created!"
    except sqlite3.IntegrityError:
        conn.close()
        return "❌ Group name already taken"

def join_group(user_id, group_name):
    if not group_name.strip():
        return "❌ Group name required"
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM groups WHERE name = ?", (group_name.strip(),))
    row = c.fetchone()
    if not row:
        conn.close()
        return "❌ Group not found"
    group_id = row[0]
    try:
        c.execute("INSERT INTO group_members (group_id, user_id) VALUES (?, ?)", (group_id, user_id))
        conn.commit()
        conn.close()
        return f"✅ Joined group '{group_name.strip()}'!"
    except sqlite3.IntegrityError:
        conn.close()
        return "❌ Already a member of this group"

def get_user_groups(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT g.name FROM groups g
        JOIN group_members gm ON gm.group_id = g.id
        WHERE gm.user_id = ?
    """, (user_id,))
    groups = [row[0] for row in c.fetchall()]
    conn.close()
    return groups

def get_group_leaderboard(group_name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM groups WHERE name = ?", (group_name,))
    row = c.fetchone()
    if not row:
        conn.close()
        return pd.DataFrame(), "❌ Group not found"
    group_id = row[0]
    df = pd.read_sql_query("""
        SELECT u.username, COUNT(b.id) as books_read
        FROM group_members gm
        JOIN users u ON u.id = gm.user_id
        LEFT JOIN books_read b ON b.user_id = u.id
        WHERE gm.group_id = ?
        GROUP BY u.id
        ORDER BY books_read DESC
    """, conn, params=(group_id,))
    conn.close()
    df.index = df.index + 1
    df.index.name = "rank"
    return df.reset_index(), None

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as app:

    user_state = gr.State(None)
    last_results_state = gr.State(None)

    gr.Markdown("# 📚 Book Recommender + Tracker")

    # ---- LOGIN SCREEN ----
    with gr.Column(visible=True) as login_screen:
        gr.Markdown("## 🔐 Login")
        login_user = gr.Textbox(label="Username")
        login_pass = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login", variant="primary")
        login_status = gr.Textbox(label="", show_label=False)
        go_register_btn = gr.Button("Don't have an account? Register here")

    # ---- REGISTER SCREEN ----
    with gr.Column(visible=False) as register_screen:
        gr.Markdown("## 📝 Create an Account")
        reg_user = gr.Textbox(label="Username")
        reg_pass = gr.Textbox(label="Password", type="password")
        reg_btn = gr.Button("Register", variant="primary")
        reg_status = gr.Textbox(label="", show_label=False)
        go_login_btn = gr.Button("Already have an account? Log in here")

    # ---- MAIN APP ----
    with gr.Column(visible=False) as main_app:
        logged_in_label = gr.Markdown("")

        with gr.Tabs() as main_tabs:
            with gr.Tab("🔍 Find Books", id=0):
                gr.Markdown("## 🔍 Find Books")
                query = gr.Textbox(label="Search", placeholder="e.g. dark fantasy with magic")
                search_btn = gr.Button("Search")
                output = gr.Dataframe()
                search_status = gr.Textbox(label="Status")
                gr.Markdown("### Add a result to your list (0 = first result)")
                index_input = gr.Number(label="Row Index", value=0)
                add_read_btn = gr.Button("✅ Mark as Read")
                add_want_btn = gr.Button("⭐ Want to Read")
                add_status = gr.Textbox(label="Status")

                search_btn.click(search, inputs=[query, user_state, last_results_state],
                                 outputs=[output, last_results_state, search_status])
                add_read_btn.click(lambda idx, u, r: add_selected_book(idx, "read", u, r),
                                   inputs=[index_input, user_state, last_results_state], outputs=add_status)
                add_want_btn.click(lambda idx, u, r: add_selected_book(idx, "want", u, r),
                                   inputs=[index_input, user_state, last_results_state], outputs=add_status)

            with gr.Tab("📝 Track Book", id=1):
                gr.Markdown("## 📝 Track a Book")
                title = gr.Textbox(label="Title")
                authors = gr.Textbox(label="Authors")
                categories = gr.Textbox(label="Categories")
                rating = gr.Textbox(label="Rating")
                read_btn = gr.Button("✅ Mark as Read")
                want_btn = gr.Button("⭐ Want to Read")
                track_status = gr.Textbox(label="Status")

                read_btn.click(track_add_read, inputs=[title, authors, categories, rating, user_state], outputs=track_status)
                want_btn.click(track_add_want, inputs=[title, authors, categories, rating, user_state], outputs=track_status)

            with gr.Tab("✅ Books Read", id=2):
                gr.Markdown("## ✅ Books Read")
                read_df_display = gr.Dataframe()
                read_remove_dd = gr.Dropdown(label="Select book to remove", choices=[])
                delete_read_btn = gr.Button("❌ Remove")
                read_status = gr.Textbox(label="Status")
                refresh_read_btn = gr.Button("🔄 Refresh")

                def refresh_read(user):
                    if not user:
                        return pd.DataFrame(), gr.Dropdown(choices=[])
                    return get_user_books(user["id"], "books_read"), gr.Dropdown(choices=get_book_titles(user["id"], "books_read"), value=None)

                def do_remove_read(title, user):
                    if not user:
                        return pd.DataFrame(), gr.Dropdown(choices=[]), "❌ Please log in first"
                    remove_book(user["id"], title, "books_read")
                    return get_user_books(user["id"], "books_read"), gr.Dropdown(choices=get_book_titles(user["id"], "books_read"), value=None), f"🗑️ Removed '{title}'"

                refresh_read_btn.click(refresh_read, inputs=user_state, outputs=[read_df_display, read_remove_dd])
                delete_read_btn.click(do_remove_read, inputs=[read_remove_dd, user_state],
                                      outputs=[read_df_display, read_remove_dd, read_status])

            with gr.Tab("⭐ Want to Read", id=3):
                gr.Markdown("## ⭐ Want to Read")
                want_df_display = gr.Dataframe()
                want_remove_dd = gr.Dropdown(label="Select book to remove", choices=[])
                delete_want_btn = gr.Button("❌ Remove")
                want_status = gr.Textbox(label="Status")
                refresh_want_btn = gr.Button("🔄 Refresh")

                def refresh_want(user):
                    if not user:
                        return pd.DataFrame(), gr.Dropdown(choices=[])
                    return get_user_books(user["id"], "books_want"), gr.Dropdown(choices=get_book_titles(user["id"], "books_want"), value=None)

                def do_remove_want(title, user):
                    if not user:
                        return pd.DataFrame(), gr.Dropdown(choices=[]), "❌ Please log in first"
                    remove_book(user["id"], title, "books_want")
                    return get_user_books(user["id"], "books_want"), gr.Dropdown(choices=get_book_titles(user["id"], "books_want"), value=None), f"🗑️ Removed '{title}'"

                refresh_want_btn.click(refresh_want, inputs=user_state, outputs=[want_df_display, want_remove_dd])
                delete_want_btn.click(do_remove_want, inputs=[want_remove_dd, user_state],
                                      outputs=[want_df_display, want_remove_dd, want_status])

            with gr.Tab("👥 Friends", id=4):
                gr.Markdown("## 👥 Friends")

                gr.Markdown("### Send a Friend Request")
                friend_input = gr.Textbox(label="Username")
                send_req_btn = gr.Button("Send Request")
                friend_status = gr.Textbox(label="Status")

                gr.Markdown("### Pending Requests")
                pending_dd = gr.Dropdown(label="Incoming requests", choices=[])
                with gr.Row():
                    accept_btn = gr.Button("✅ Accept")
                    decline_btn = gr.Button("❌ Decline")
                request_status = gr.Textbox(label="Status")
                refresh_requests_btn = gr.Button("🔄 Refresh Requests")

                gr.Markdown("### Your Friends")
                friends_list = gr.Dropdown(label="Select a friend to view their books", choices=[])
                refresh_friends_btn = gr.Button("🔄 Refresh Friends")
                view_friend_btn = gr.Button("View Selected Friend's Books")
                friend_read_display = gr.Dataframe(label="Books Read")
                friend_want_display = gr.Dataframe(label="Want to Read")
                view_status = gr.Textbox(label="Status")

                def do_send_request(friend_username, user):
                    if not user:
                        return "❌ Please log in first"
                    return send_friend_request(user["id"], friend_username)

                def do_refresh_requests(user):
                    if not user:
                        return gr.Dropdown(choices=[])
                    reqs = get_pending_requests(user["id"])
                    choices = [f"{req_id}:{username}" for req_id, username in reqs]
                    return gr.Dropdown(choices=choices, value=None)

                def do_accept(selected, user):
                    if not user or not selected:
                        return "❌ Select a request first", gr.Dropdown(choices=[])
                    req_id = int(selected.split(":")[0])
                    msg = accept_request(req_id, user["id"])
                    reqs = get_pending_requests(user["id"])
                    choices = [f"{r}:{u}" for r, u in reqs]
                    return msg, gr.Dropdown(choices=choices, value=None)

                def do_decline(selected, user):
                    if not user or not selected:
                        return "❌ Select a request first", gr.Dropdown(choices=[])
                    req_id = int(selected.split(":")[0])
                    msg = decline_request(req_id, user["id"])
                    reqs = get_pending_requests(user["id"])
                    choices = [f"{r}:{u}" for r, u in reqs]
                    return msg, gr.Dropdown(choices=choices, value=None)

                def do_refresh_friends(user):
                    if not user:
                        return gr.Dropdown(choices=[])
                    return gr.Dropdown(choices=get_friends(user["id"]), value=None)

                def do_view_friend(friend_username, user):
                    if not user:
                        return pd.DataFrame(), pd.DataFrame(), "❌ Please log in first"
                    if not friend_username:
                        return pd.DataFrame(), pd.DataFrame(), "❌ Select a friend first"
                    read_df, err = get_friend_books(user["id"], friend_username, "books_read")
                    if err:
                        return pd.DataFrame(), pd.DataFrame(), err
                    want_df, err = get_friend_books(user["id"], friend_username, "books_want")
                    if err:
                        return pd.DataFrame(), pd.DataFrame(), err
                    return read_df, want_df, f"📖 Showing {friend_username}'s books"

                send_req_btn.click(do_send_request, inputs=[friend_input, user_state], outputs=friend_status)
                refresh_requests_btn.click(do_refresh_requests, inputs=user_state, outputs=pending_dd)
                accept_btn.click(do_accept, inputs=[pending_dd, user_state], outputs=[request_status, pending_dd])
                decline_btn.click(do_decline, inputs=[pending_dd, user_state], outputs=[request_status, pending_dd])
                refresh_friends_btn.click(do_refresh_friends, inputs=user_state, outputs=friends_list)
                view_friend_btn.click(do_view_friend, inputs=[friends_list, user_state],
                                      outputs=[friend_read_display, friend_want_display, view_status])

            with gr.Tab("🏆 Leaderboard", id=5):
                gr.Markdown("## 🏆 Leaderboard")
                leaderboard_display = gr.Dataframe(label="Global Leaderboard — Books Read")
                refresh_lb_btn = gr.Button("🔄 Refresh")
                refresh_lb_btn.click(get_global_leaderboard, outputs=leaderboard_display)

            with gr.Tab("🎯 Groups", id=6):
                gr.Markdown("## 🎯 Groups")
                gr.Markdown("### Create or Join a Group")
                group_name_input = gr.Textbox(label="Group name")
                with gr.Row():
                    create_group_btn = gr.Button("Create Group")
                    join_group_btn = gr.Button("Join Group")
                group_action_status = gr.Textbox(label="Status")

                gr.Markdown("### Your Groups")
                my_groups_dd = gr.Dropdown(label="Select a group", choices=[])
                refresh_groups_btn = gr.Button("🔄 Refresh My Groups")
                view_group_lb_btn = gr.Button("View Group Leaderboard")
                group_lb_display = gr.Dataframe(label="Group Leaderboard")
                group_lb_status = gr.Textbox(label="Status")

                def do_create_group(name, user):
                    if not user:
                        return "❌ Please log in first", gr.Dropdown(choices=[])
                    msg = create_group(user["id"], name)
                    return msg, gr.Dropdown(choices=get_user_groups(user["id"]), value=None)

                def do_join_group(name, user):
                    if not user:
                        return "❌ Please log in first", gr.Dropdown(choices=[])
                    msg = join_group(user["id"], name)
                    return msg, gr.Dropdown(choices=get_user_groups(user["id"]), value=None)

                def do_refresh_groups(user):
                    if not user:
                        return gr.Dropdown(choices=[])
                    return gr.Dropdown(choices=get_user_groups(user["id"]), value=None)

                def do_view_group_lb(group_name):
                    if not group_name:
                        return pd.DataFrame(), "❌ Select a group first"
                    df, err = get_group_leaderboard(group_name)
                    if err:
                        return pd.DataFrame(), err
                    return df, f"🏆 {group_name} leaderboard"

                create_group_btn.click(do_create_group, inputs=[group_name_input, user_state],
                                       outputs=[group_action_status, my_groups_dd])
                join_group_btn.click(do_join_group, inputs=[group_name_input, user_state],
                                     outputs=[group_action_status, my_groups_dd])
                refresh_groups_btn.click(do_refresh_groups, inputs=user_state, outputs=my_groups_dd)
                view_group_lb_btn.click(do_view_group_lb, inputs=my_groups_dd,
                                        outputs=[group_lb_display, group_lb_status])

            with gr.Tab("⚙️ Settings", id=7):
                gr.Markdown("## ⚙️ Settings")

                gr.Markdown("### Change Password")
                current_pass = gr.Textbox(label="Current Password", type="password")
                new_pass = gr.Textbox(label="New Password", type="password")
                change_pass_btn = gr.Button("Update Password")
                change_pass_status = gr.Textbox(label="Status")

                gr.Markdown("### Account")
                logout_btn = gr.Button("🚪 Log Out", variant="stop")

                def do_change_password(current, new, user):
                    if not user:
                        return "❌ Please log in first"
                    return change_password(user["id"], current, new)

                def do_logout():
                    return (None, gr.update(visible=True), gr.update(visible=False),
                            gr.update(visible=False), "")

                change_pass_btn.click(do_change_password,
                                      inputs=[current_pass, new_pass, user_state],
                                      outputs=change_pass_status)
                logout_btn.click(do_logout,
                                 outputs=[user_state, login_screen, register_screen, main_app, logged_in_label])

    # ---- AUTH SCREEN LOGIC ----
    def do_login(username, password):
        user, msg = login(username, password)
        if user:
            return (user, gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True), f"**Logged in as: {user['username']}**", "",
                    gr.update(selected=0))
        return None, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", msg, gr.update()

    def do_register(username, password):
        user, msg = register(username, password)
        if user:
            return (user, gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True), f"**Logged in as: {user['username']}**", "",
                    gr.update(selected=0))
        return None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "", msg, gr.update()

    auth_outputs = [user_state, login_screen, register_screen, main_app, logged_in_label, login_status, main_tabs]
    reg_outputs  = [user_state, login_screen, register_screen, main_app, logged_in_label, reg_status, main_tabs]

    login_btn.click(do_login, inputs=[login_user, login_pass], outputs=auth_outputs)
    reg_btn.click(do_register, inputs=[reg_user, reg_pass], outputs=reg_outputs)

    go_register_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)),
                          outputs=[login_screen, register_screen])
    go_login_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)),
                       outputs=[login_screen, register_screen])

app.launch()
