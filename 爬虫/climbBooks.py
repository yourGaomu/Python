#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爬虫: 从 http://books.toscrape.com/ 抓取书籍信息并保存为 JSON/CSV，最后做简单分析。
提取字段：Title, Price, Availability, Category, Description, Rating, UPC, ProductPageURL。
"""

import time
import csv
import json
import re
from typing import List, Dict
from urllib.parse import urljoin

import requests
from lxml import etree

BASE = "http://books.toscrape.com/"
START_URL = urljoin(BASE, "catalogue/page-1.html")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

RATE_LIMIT = 0.1  # 每请求间隔
DEFAULT_MAX_BOOKS = 100  # 默认抓取前 100 本
CSV_COLUMNS = [
    "Title",
    "Price",
    "Availability",
    "Category",
    "Description",
    "Rating",
    "UPC",
    "ProductPageURL",
]


def fetch(url: str, session: requests.Session, timeout: float = 10.0, retries: int = 2) -> etree._Element:
    """Fetch URL with timeout and simple retry. Returns parsed tree or None on failure."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            html = resp.content
            return etree.HTML(html)
        except requests.RequestException as e:
            last_err = e
            print(f"请求失败（第 {attempt+1}/{retries+1} 次）{url}: {e}")
            time.sleep(0.5)
            continue
    print(f"最终失败: {url} -> {last_err}")
    return None


def parse_list_page(tree: etree._Element) -> List[str]:
    """返回详情页相对链接列表（相对于 BASE 或 catalogue）"""
    links = []
    # 每本书的 h3/a @href
    for a in tree.xpath("//article[@class='product_pod']/h3/a/@href"):
        links.append(a)
    return links


STAR_MAP = {
    'One': 1,
    'Two': 2,
    'Three': 3,
    'Four': 4,
    'Five': 5
}


def parse_detail_page(tree: etree._Element, page_url: str) -> Dict:
    # title
    raw_title = tree.xpath("string(//div[contains(@class,'product_main')]/h1)") or ""
    title = raw_title.strip() or "N/A"

    # price (带货币符号，例 £51.77)
    price_text = tree.xpath("string(//p[@class='price_color'])") or ""
    price_match = re.search(r"[\d.]+", price_text)
    price_value = float(price_match.group()) if price_match else 0.0

    # availability -> 提取数字
    availability_text = tree.xpath("string(//p[@class='instock availability'])") or ""
    availability_text = ' '.join(availability_text.split())
    availability_match = re.search(r"\d+", availability_text)
    availability_count = int(availability_match.group()) if availability_match else 0

    # category: breadcrumb third li
    category_text = tree.xpath("string(//ul[@class='breadcrumb']/li[3]/a)") or ""
    category = category_text.strip() or "N/A"

    # product description: next p after #product_description h2
    desc = "N/A"
    desc_nodes = tree.xpath("//div[@id='product_description']/following-sibling::p[1]/text()")
    if desc_nodes:
        desc_candidate = desc_nodes[0].strip()
        if desc_candidate:
            desc = desc_candidate

    # star rating: class on <p class="star-rating Three">
    star_cls = tree.xpath("//p[contains(@class,'star-rating')]/@class")
    star = 0
    if star_cls:
        parts = star_cls[0].split()
        for p in parts:
            if p in STAR_MAP:
                star = STAR_MAP[p]
                break

    # UPC: table where th text is 'UPC'
    upc = "N/A"
    upc_text = tree.xpath("string(//table//tr[th/text()='UPC']/td)")
    if upc_text:
        upc_candidate = upc_text.strip()
        if upc_candidate:
            upc = upc_candidate

    return {
        'Title': title,
        'Price': price_value,
        'Availability': availability_count,
        'Category': category,
        'Description': desc,
        'Rating': star,
        'UPC': upc,
        'ProductPageURL': page_url
    }


def crawl(start_url: str = START_URL, max_pages: int = 0, max_books: int = 0) -> List[Dict]:
    session = requests.Session()
    session.headers.update(HEADERS)
    results = []

    # iterate pagination
    next_page = start_url
    page_count = 0
    while next_page:
        page_count += 1
        if max_pages and page_count > max_pages:
            print(f"到达最大页面限制 {max_pages}，停止翻页")
            break
        print(f"正在抓取列表页: {next_page}")
        tree = fetch(next_page, session)
        if tree is None:
            print(f"无法获取列表页，跳过: {next_page}")
            break
        links = parse_list_page(tree)
        # links are relative; some start without ../ so join with page base
        for href in links:
            # normalize to absolute: many links like '../../../...' relative to catalogue
            detail_url = urljoin(next_page, href)
            print(f"  -> 抓取详情页: {detail_url}")
            dtree = fetch(detail_url, session)
            if dtree is None:
                print(f"跳过详情页 (获取失败): {detail_url}")
                time.sleep(RATE_LIMIT)
                continue
            info = parse_detail_page(dtree, detail_url)
            results.append(info)
            time.sleep(RATE_LIMIT)
            if max_books and len(results) >= max_books:
                print(f"达到抓取数量上限 {max_books} 本，停止")
                return results
        # find next page
        next_rel = tree.xpath("//li[@class='next']/a/@href")
        if next_rel:
            next_page = urljoin(next_page, next_rel[0])
            time.sleep(RATE_LIMIT)
        else:
            next_page = None
    return results


def save_json(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(data: List[Dict], path: str):
    if not data:
        return
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in data:
            writer.writerow({col: row.get(col, "N/A") for col in CSV_COLUMNS})


def analyze(data: List[Dict]) -> Dict:
    # 基本分析：平均价格、热门分类、最高价格书目、评分分布
    from collections import Counter
    cat_counter = Counter()
    rating_counter = Counter()
    price_sum = 0.0
    price_count = 0

    for item in data:
        category = item.get('Category', 'N/A') or 'N/A'
        cat_counter[category] += 1
        rating_counter[item.get('Rating', 0)] += 1
        price = item.get('Price', 0.0)
        if isinstance(price, (int, float)):
            price_sum += price
            price_count += 1

    avg_price = price_sum / price_count if price_count else 0.0
    top_price_books = sorted(data, key=lambda x: x.get('Price', 0.0), reverse=True)[:5]
    top_categories = cat_counter.most_common(3)

    return {
        'top_price_books': top_price_books,
        'top_categories': top_categories,
        'rating_counts': dict(sorted(rating_counter.items())),
        'average_price': round(avg_price, 2),
        'total_books': len(data)
    }


def main(max_books: int = DEFAULT_MAX_BOOKS):
    data = crawl(max_books=max_books)
    print(f"共抓取 {len(data)} 本图书")
    save_json(data, 'books.json')
    save_csv(data, 'books.csv')
    stats = analyze(data)
    stats_cn = {
        "最高价格前五本": [
            {
                "标题": book.get("Title", "N/A"),
                "价格": book.get("Price", 0.0),
                "分类": book.get("Category", "N/A"),
            }
            for book in stats["top_price_books"]
        ],
        "最热门的三个分类": stats["top_categories"],
        "评分统计": stats["rating_counts"],
        "平均价格": stats["average_price"],
        "图书总数": stats["total_books"],
    }
    print("数据分析：")
    print(json.dumps(stats_cn, ensure_ascii=False, indent=2))
    with open('analysis.json', 'w', encoding='utf-8') as f:
        json.dump(stats_cn, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
