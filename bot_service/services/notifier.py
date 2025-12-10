from typing import Optional
from io import BytesIO
import config
from telegram import Bot


class Notifier:
    def __init__(self, bot: Bot, chat_id: str):
        self.bot = bot
        self.chat_id = chat_id

    async def send_text(self, text: str, parse_mode: Optional[str] = None):
        await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=parse_mode)

    async def send_photo(self, image_path: str, caption: Optional[str] = None, parse_mode: Optional[str] = None):
        with open(image_path, 'rb') as f:
            await self.bot.send_photo(chat_id=self.chat_id, photo=f, caption=caption, parse_mode=parse_mode)
