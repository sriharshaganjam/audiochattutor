import os
print(os.getcwd())
import edge_tts, asyncio
async def test():
    communicate = edge_tts.Communicate("Hello world", "en-US-JennyNeural")
    await communicate.save("test.mp3")
asyncio.run(test())