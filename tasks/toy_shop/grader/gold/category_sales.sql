WITH lines AS (
    SELECT it.product_id AS product_id, it.quantity AS quantity
    FROM (SELECT status, UNNEST(items) AS it FROM {{ source('elt', 'orders') }})
    WHERE status <> 'cancelled'
),
prod_rev AS (
    SELECT p.product_id, p.product_name, p.category,
           COALESCE(SUM(l.quantity), 0) AS units,
           COALESCE(SUM(l.quantity * p.unit_price), 0) AS rev
    FROM {{ source('elt', 'products') }} p
    LEFT JOIN lines l USING (product_id)
    GROUP BY p.product_id, p.product_name, p.category
),
top AS (
    SELECT category, product_name,
           ROW_NUMBER() OVER (PARTITION BY category ORDER BY rev DESC, product_name ASC) AS rn
    FROM prod_rev
    WHERE units > 0
)
SELECT pr.category,
       COUNT(*) AS num_products,
       SUM(pr.units) AS units_sold,
       ROUND(SUM(pr.rev), 2) AS revenue,
       t.product_name AS top_product
FROM prod_rev pr
LEFT JOIN top t ON t.category = pr.category AND t.rn = 1
GROUP BY pr.category, t.product_name
