import json


def load(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def unique_users(data: dict) -> dict:
    messages = data['messages']
    users = {}
    for msg in messages:
        if 'from' not in msg:
            continue
        users[msg['from_id']] = msg['from']
    return users


def user_message_texts(data: dict, user_id: str) -> str:
    messages = data['messages']
    result = []
    for msg in messages:
        if (
                'forwarded_from' in msg
                or msg.get('from_id', '') != user_id
                or isinstance(msg['text'], list)
        ):
            continue
        if msg['text']:
            result.append(msg['text'])
    return '. '.join(result)


def save_text(text: str, user_id: str):
    with open(f'{user_id}.txt', 'w') as f:
        f.write(text)
    print('Successfully saved!')


if __name__ == '__main__':
    data = load('result.json')
    print(unique_users(data))
    text = user_message_texts(data, 'user229875949')
    save_text(text, 'user229875949')
