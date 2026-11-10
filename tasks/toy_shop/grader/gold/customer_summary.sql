WITH lines AS (
    SELECT order_id, customer_id, it.product_id AS product_id, it.quantity AS quantity
    FROM (SELECT order_id, customer_id, status, UNNEST(items) AS it FROM {{ source('elt', 'orders') }})
    WHERE status <> 'cancelled'
),
priced AS (
    SELECT l.*, p.category, l.quantity * p.unit_price AS amount
    FROM lines l
    JOIN {{ source('elt', 'products') }} p USING (product_id)
),
agg AS (
    SELECT customer_id, COUNT(DISTINCT order_id) AS num_orders, ROUND(SUM(amount), 2) AS total_spent
    FROM priced
    GROUP BY customer_id
),
cat_rank AS (
    SELECT customer_id, category,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY SUM(quantity) DESC, category ASC) AS rn
    FROM priced
    GROUP BY customer_id, category
)
SELECT c.customer_id,
       c.first_name || ' ' || c.last_name AS customer_name,
       c.country,
       COALESCE(a.num_orders, 0) AS num_orders,
       COALESCE(a.total_spent, 0) AS total_spent,
       r.category AS favorite_category,
       CASE WHEN COALESCE(a.total_spent, 0) > 100 THEN 1 ELSE 0 END AS is_big_spender
FROM {{ source('elt', 'customers') }} c
LEFT JOIN agg a USING (customer_id)
LEFT JOIN cat_rank r ON r.customer_id = c.customer_id AND r.rn = 1
