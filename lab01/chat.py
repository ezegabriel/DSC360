'''
Gabriel Eze
Lab-01
DSC 360
Building a chatbot
'''

import ollama
import os
from datetime import datetime

# Locate base path
TRANSCRIPTS_BASE = 'transcripts'
MODEL = 'gemma3:4b'

# Time of conversation log
def iso_now():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

# Text file to save conversation log
def session_filename():
    return datetime.now().strftime('%Y%m%d_%H%M%S') + '.txt'



def main():
    print('Welcome to Lab01 chatbot! Type /help for commands, /exit to quit.\n')
    

    # Ensure a daily session subfolder exists
    sesh_dir = os.path.join(TRANSCRIPTS_BASE, datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(sesh_dir, exist_ok = True)

    # Open new session file
    filepath = os.path.join(sesh_dir, session_filename())
    transcript = open(filepath, 'a', encoding = 'utf-8')
    print(f'(Saving transcript to: {filepath})\n')
    
    model = MODEL
    conversation = []
    try:
        while True:
            user_inpt = input('You: ').strip()
            print()

            # Commands
            if user_inpt.lower() == '/exit':
                print('Goodbye!')
                break

            elif user_inpt.lower() == '/help':
                print('\nAvailable commands:')
                print('/exit    - quit the chatbot')
                print('/new     - clear conversation history')
                print('/model <name>   - switch model in-session')
                print('/help    - show this help message\n')
                continue

            elif user_inpt.lower() == '/new':
                conversation = []
                print('Conversation history cleared. Starting afresh!\n')
                continue
            
            elif '/model' in user_inpt.lower():
                model = user_inpt[user_inpt.find(' ') + 1:]               
                continue

            # Log user line immediately
            transcript.write(f'[{iso_now()}][user] {user_inpt}\n\n')
            transcript.flush()  


            try:
                conversation.append({'role': 'user', 'content': user_inpt})

                # Show model name, then stream content matter
                print(f'{model}: ', end = '', flush = True)

                stream = ollama.chat(model = model, messages = conversation, stream = True)
                reply_parts = []

                for chunk in stream:
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        print(content, end = '', flush = True)
                        reply_parts.append(content)

                print('\n')

                # Gather full reply
                reply = ''.join(reply_parts)
        

                conversation.append({'role': 'assistant', 'content': reply})

                # Log assistant line immediately
                transcript.write(f'[{iso_now()}][{model}] {reply}\n\n')
                transcript.flush()

            except Exception as e:
                err_msg = f'[{iso_now()}][system] Error: {e}'
                print(err_msg + '\n')
                transcript.write(err_msg + '\n\n')
                transcript.flush()
    finally:
        transcript.close()


if __name__ == '__main__':
    main()
