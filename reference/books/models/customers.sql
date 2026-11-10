-- Reference model for ELT-Bench books.customers (see data_model.yaml).
with orders as (
    select order_id, customer_id, shipping_method_id from cust_order
),
methods as (
    select try_to_number(method_id) as method_id, method_name from shipping_method
),
oldest_books as (
    select cast(book_id as int) as book_id from book
    where try_to_date(publication_date) = (select min(try_to_date(publication_date)) from book)
),
lines as (
    select o.customer_id, ol.book_id, ol.price
    from order_line ol
    join orders o on o.order_id = ol.order_id
),
per_customer_books as (
    select customer_id,
           100.0 * count_if(price > 13) / count(*) as per_over_13,
           max(iff(book_id in (select book_id from oldest_books), 1, 0)) as has_oldest
    from lines
    group by customer_id
),
per_customer_orders as (
    select o.customer_id,
           count(*) as num_orders,
           100.0 * count_if(m.method_name = 'International') / count(*) as per_international
    from orders o
    left join methods m on m.method_id = o.shipping_method_id
    group by o.customer_id
),
preferred as (
    select o.customer_id, m.method_name
    from orders o
    join methods m on m.method_id = o.shipping_method_id
    group by o.customer_id, m.method_name
    qualify row_number() over (partition by o.customer_id order by count(*) desc, m.method_name asc) = 1
)
select cast(c.customer_id as int) as customer_id,
       c.first_name || ' ' || c.last_name as customer_name,
       c.email,
       b.per_over_13 as per_books_over_13,
       coalesce(o.num_orders, 0) as num_orders,
       p.method_name as preferred_shipping_method,
       coalesce(b.has_oldest, 0) as has_ordered_the_oldest_book,
       o.per_international as per_order_shipped_internationally
from customer c
left join per_customer_books b on b.customer_id = cast(c.customer_id as int)
left join per_customer_orders o on o.customer_id = cast(c.customer_id as int)
left join preferred p on p.customer_id = cast(c.customer_id as int)
