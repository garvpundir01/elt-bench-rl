-- Reference model for ELT-Bench books.publishers (see data_model.yaml).
with b as (
    select cast(book_id as int) as book_id,
           title,
           num_pages,
           try_to_date(publication_date) as pub_date,
           cast(publisher_id as int) as publisher_id,
           cast(language_id as int) as language_id
    from book
),
avg_pages as (
    select avg(num_pages) as a from b
),
per_publisher as (
    select b.publisher_id,
           count(*) as num_books,
           count_if(b.num_pages > 0.7 * (select a from avg_pages)) as n_over_70pct,
           100.0 * count_if(l.language_name = 'Japanese') / count(*) as per_japanese
    from b
    left join book_language l on cast(l.language_id as int) = b.language_id
    group by b.publisher_id
),
oldest as (
    select publisher_id, title from b
    qualify row_number() over (partition by publisher_id order by pub_date asc, title asc) = 1
),
most_pages as (
    select publisher_id, title from b
    qualify row_number() over (partition by publisher_id order by num_pages desc, title desc) = 1
)
select cast(p.publisher_id as int) as publisher_id,
       p.publisher_name,
       coalesce(pp.num_books, 0) as num_books,
       coalesce(pp.n_over_70pct, 0) as num_books_pages_greater_than_70_per_of_avg,
       pp.per_japanese as per_japanese_books,
       o.title as title_of_the_oldest_book,
       m.title as book_title_with_the_most_pages
from publisher p
left join per_publisher pp on pp.publisher_id = cast(p.publisher_id as int)
left join oldest o on o.publisher_id = cast(p.publisher_id as int)
left join most_pages m on m.publisher_id = cast(p.publisher_id as int)
