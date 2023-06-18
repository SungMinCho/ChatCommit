import os

import git
import openai
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-3.5-turbo"

openai.api_key = OPENAI_API_KEY


SYSTEM_PROMPT = """I want you to act as a commit message generator.
I will provide you with the local code diff,
and I would like you to generate an appropriate commit message using the conventional commit format.
You should also include helpful prefixes in commit message such as [feat], [chore], [fix], etc...
Do not write any explanations or other words, just reply with the commit message.
The user might want to work with you to iteratively refine and improve the commit message.
I want you to reflect the user's intentions at every request.
However, you must only reply with the commit message at all times.
"""


class Chat:
    def __init__(self, system_prompt=SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": system_prompt}]

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL_NAME,
            messages=self.history,
        )

        ret = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": ret})

        print("Chat Response", ret)

        return ret

    def generate_history(self):
        return "\n".join(
            f"{h['role'].capitalize()}: {h['content']}" for h in self.history
        )


chatbot = Chat()
repo_chatbot = {}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "repo_path" in request.form:
            repo_path = request.form["repo_path"]
            if os.path.isdir(repo_path):
                try:
                    repo = git.Repo(repo_path)
                    commits = list(repo.iter_commits())
                    commit_branches = get_commit_branches(repo)
                    diff = get_local_code_diff(repo)

                    if repo_path not in repo_chatbot:
                        chat = Chat()
                        repo_chatbot[repo_path] = chat
                        # first query
                        chat_response = chat.chat(diff)
                    elif "chat_message" in request.form:
                        chat = repo_chatbot[repo_path]
                        chat_response = chat.chat(request.form["chat_message"])
                    else:
                        chat_response = None

                    return render_template(
                        "app.html",
                        repo_path=repo_path,
                        commits=commits,
                        commit_branches=commit_branches,
                        diff=diff,
                        generated_commit=chat_response,
                        chat_response=chat_response,
                    )

                except git.InvalidGitRepositoryError:
                    error_message = "Invalid Git repository"
            else:
                error_message = "Invalid directory"

            return render_template("app.html", error_message=error_message)

    return render_template("app.html")


@app.route("/chat_respond", methods=["POST"])
def chat_respond():
    repo_path = request.form["repo_path"]
    chat = repo_chatbot[repo_path]
    chat_message = request.form["chat_message"]
    response = chat.chat(chat_message)
    return response


def get_commit_branches(repo):
    commit_branches = {}
    for branch in repo.branches:
        commit_branches[branch.name] = [
            commit.hexsha for commit in repo.iter_commits(branch.name)
        ]
    return commit_branches


def get_local_code_diff(repo):
    diff = ""
    if repo.head.is_valid():
        diff = repo.git.diff(repo.head.commit.tree)
    return diff


def process_chat_message(chat_message, diff):
    prompt = f"## Code Diff:\n\n{diff}\n\n## Chat Message:\n\n{chat_message}\n\n## Generated Commit Message:"

    # Make API call to OpenAI with the appropriate prompt and content of the code diff
    response = requests.post(
        f"https://api.openai.com/v1/engines/{OPENAI_MODEL_NAME}/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "prompt": prompt,
            "max_tokens": 64,
        },
    )

    if response.status_code == 200:
        data = response.json()
        completions = data["choices"]
        if completions:
            commit_message = completions[0]["text"].strip()
            return commit_message

    return "Error generating commit message"


if __name__ == "__main__":
    app.run(debug=True, port=4300)
