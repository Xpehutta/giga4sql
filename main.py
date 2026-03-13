from dotenv import load_dotenv
import os
GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS")

load_dotenv()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    str_ = True if GIGACHAT_CREDENTIALS else False
    print_hi(str_)


