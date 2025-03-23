import ollama
import pprint

def main():
    image = "rc_car_dataset/images/F1_0124.jpg"
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': 'Tell me about this image?',
            'images': [image]
        }]
    )

    print(response.get('message').get('content'))
    return response.get('message').get('content')

if __name__ == '__main__':
    main()