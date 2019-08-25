from html.parser import HTMLParser


class ForexNewsParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.dates = set()

    def error(self, message):
        print("error: ", message)

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            for attr in attrs:
                if attr[0] == 'data-event-datetime':
                    self.dates.add(attr[1])


if __name__ == '__main__':
    with open("./data/Economic Calendar - Investing.com.html", 'r', encoding='utf-8') as f:
        html_raw = f.read()
    parser = ForexNewsParser()
    parser.feed(html_raw)
    print(sorted(parser.dates))
    print(len(parser.dates))