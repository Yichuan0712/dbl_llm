from bs4 import BeautifulSoup


def extract_article_content(html):
    soup = BeautifulSoup(html, 'html.parser')

    article_section = soup.find('section', {'aria-label': 'Article content'})

    if article_section:
        return '\n'.join([p.get_text() for p in article_section.find_all('p')])
    else:
        return "NOT FOUND"
