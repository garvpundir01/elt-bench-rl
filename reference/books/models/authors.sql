-- Reference model for ELT-Bench books.authors (see data_model.yaml).
with books as (
    select cast(book_id as int) as book_id, title, num_pages, try_to_date(publication_date) as pub_date
    from book
),
author_books as (
    select cast(ba.author_id as int) as author_id, b.*
    from book_author ba
    join books b on b.book_id = cast(ba.book_id as int)
),
first_book as (
    select author_id, title from author_books
    qualify row_number() over (partition by author_id order by pub_date asc, title asc) = 1
),
stats as (
    select author_id,
           100.0 * count_if(year(pub_date) = 1992) / count(*) as per_1992,
           sum(num_pages) as total_pages
    from author_books
    group by author_id
),
avg_total_pages as (
    select avg(total_pages) as a from stats
),
sales as (
    select ab.author_id, ab.title, ol.price
    from author_books ab
    join order_line ol on ol.book_id = ab.book_id
),
most_expensive as (
    select author_id, title from sales
    qualify row_number() over (partition by author_id order by price desc, title desc) = 1
),
avg_price as (
    select author_id, avg(price) as avg_price from sales group by author_id
)
select cast(a.author_id as int) as author_id,
       a.author_name,
       f.title as first_book,
       s.per_1992 as per_book_published_1992,
       iff(s.total_pages < (select a from avg_total_pages), 1, 0) as wrote_fewer_total_pages_than_the_average,
       m.title as most_expensive_book,
       iff(p.avg_price > 19, 1, 0) as avg_book_price_over_19
from author a
left join first_book f on f.author_id = cast(a.author_id as int)
left join stats s on s.author_id = cast(a.author_id as int)
left join most_expensive m on m.author_id = cast(a.author_id as int)
left join avg_price p on p.author_id = cast(a.author_id as int)
