import requests
from lxml import etree
from typing import Optional, Dict, List




def main(query: str, session: requests.Session) -> None:
    url = f"https://www.bing.com/search?q={query}"
    response = session.get(url)
    if response.status_code == 200:
        parser = etree.HTMLParser()
        tree = etree.fromstring(response.content, parser)
        results = tree.xpath("//li[@class='b_algo']")
        for result in results:
            title = result.xpath(".//h2/a/text()")
            link = result.xpath(".//h2/a/@href")
            summary = result.xpath(".//p/text()")
            print(f"标题: {title[0] if title else '无标题'}")
            print(f"链接: {link[0] if link else '无链接'}")
            print(f"摘要: {summary[0] if summary else '无摘要'}")
    else:
        print(f"请求失败，状态码: {response.status_code}")

if __name__ == "__main__":
    print("初始化爬虫模块")
    session = requests.Session()
    headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate",  # 移除 br, zstd 以避免解码问题
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "cache-control": "max-age=0",
            "priority": "u=0, i",
            "referer": "https://link.csdn.net/?from_id=109739116&target=https%3A%2F%2Fmovie.douban.com%2Ftop250",
            "sec-ch-ua": '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": '"Android"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "cross-site",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Mobile Safari/537.36 Edg/142.0.0.0"
        }
    session.headers.update(headers)
    # 调用主程序
    # 要求用户输入查询的问题
    query = input("请输入查询的问题: ")
    main("验证码",session)