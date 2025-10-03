# import openai
# from app.utils.config import OPENAI_API_KEY
#
# openai.api_key = OPENAI_API_KEY
#
# class LLMService:
#     def __init__(self, model: str = "text-davinci-003"):
#         self.model = model
#
#     def generate(self, prompt: str, max_tokens: int = 100) -> str:
#         response = openai.Completion.create(
#             engine=self.model,
#             prompt=prompt,
#             max_tokens=max_tokens
#         )
#         return response.choices[0].text.strip()
