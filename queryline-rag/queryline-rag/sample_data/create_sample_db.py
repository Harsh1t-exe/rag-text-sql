"""
Creates a richer sample SQLite database (an online store + support desk +
marketing) so schema-retrieval RAG has enough tables to meaningfully narrow
down from. With only 3-4 tables, RAG can't show its value - the whole
schema fits in a prompt anyway. With 10, it does.

Run: python sample_data/create_sample_db.py
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "sample_store.db")

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    country TEXT,
    signup_date DATE,
    referral_source TEXT
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price REAL NOT NULL,
    cost REAL,
    stock_quantity INTEGER
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    status TEXT,
    shipping_country TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    payment_method TEXT,
    paid_at DATETIME,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE support_tickets (
    ticket_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    subject TEXT,
    status TEXT,
    priority TEXT,
    created_at DATETIME,
    resolved_at DATETIME,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE support_agents (
    agent_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT
);

CREATE TABLE ticket_assignments (
    assignment_id INTEGER PRIMARY KEY,
    ticket_id INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,
    assigned_at DATETIME,
    FOREIGN KEY (ticket_id) REFERENCES support_tickets(ticket_id),
    FOREIGN KEY (agent_id) REFERENCES support_agents(agent_id)
);

CREATE TABLE marketing_campaigns (
    campaign_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    channel TEXT,
    start_date DATE,
    end_date DATE,
    budget REAL
);

CREATE TABLE campaign_conversions (
    conversion_id INTEGER PRIMARY KEY,
    campaign_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    converted_at DATETIME,
    FOREIGN KEY (campaign_id) REFERENCES marketing_campaigns(campaign_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

SAMPLE_DATA = """
INSERT INTO customers (customer_id, name, email, country, signup_date, referral_source) VALUES
(1, 'Aisha Khan', 'aisha@example.com', 'India', '2023-01-15', 'google_ads'),
(2, 'Liam Brown', 'liam@example.com', 'USA', '2023-03-22', 'referral'),
(3, 'Mei Chen', 'mei@example.com', 'China', '2023-05-10', 'organic'),
(4, 'Carlos Diaz', 'carlos@example.com', 'Mexico', '2023-07-01', 'instagram'),
(5, 'Sara Lee', 'sara@example.com', 'South Korea', '2023-09-18', 'google_ads'),
(6, 'Noah Wilson', 'noah@example.com', 'USA', '2024-01-05', 'organic'),
(7, 'Fatima Ali', 'fatima@example.com', 'UAE', '2024-02-14', 'referral');

INSERT INTO products (product_id, name, category, price, cost, stock_quantity) VALUES
(1, 'Wireless Mouse', 'Electronics', 19.99, 8.00, 150),
(2, 'Mechanical Keyboard', 'Electronics', 89.99, 40.00, 60),
(3, 'Yoga Mat', 'Fitness', 24.50, 10.00, 200),
(4, 'Water Bottle', 'Fitness', 12.00, 4.00, 300),
(5, 'Desk Lamp', 'Home', 34.99, 15.00, 90),
(6, 'Standing Desk', 'Home', 249.00, 120.00, 25);

INSERT INTO orders (order_id, customer_id, order_date, status, shipping_country) VALUES
(1, 1, '2024-01-05', 'completed', 'India'),
(2, 2, '2024-01-10', 'completed', 'USA'),
(3, 1, '2024-02-14', 'completed', 'India'),
(4, 3, '2024-03-01', 'pending', 'China'),
(5, 4, '2024-03-15', 'completed', 'Mexico'),
(6, 5, '2024-04-02', 'cancelled', 'South Korea'),
(7, 6, '2024-04-20', 'completed', 'USA'),
(8, 7, '2024-05-01', 'completed', 'UAE'),
(9, 2, '2024-05-12', 'completed', 'USA');

INSERT INTO order_items (order_item_id, order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 2, 19.99), (2, 1, 2, 1, 89.99), (3, 2, 3, 1, 24.50),
(4, 3, 5, 1, 34.99), (5, 4, 4, 3, 12.00), (6, 5, 2, 1, 89.99),
(7, 5, 1, 1, 19.99), (8, 6, 3, 2, 24.50), (9, 7, 6, 1, 249.00),
(10, 8, 4, 2, 12.00), (11, 9, 2, 1, 89.99);

INSERT INTO payments (payment_id, order_id, amount, payment_method, paid_at) VALUES
(1, 1, 129.97, 'card', '2024-01-05 10:00:00'),
(2, 2, 24.50, 'paypal', '2024-01-10 14:30:00'),
(3, 3, 34.99, 'card', '2024-02-14 09:15:00'),
(4, 5, 109.98, 'card', '2024-03-15 16:20:00'),
(5, 7, 249.00, 'paypal', '2024-04-20 11:00:00'),
(6, 8, 24.00, 'card', '2024-05-01 13:45:00'),
(7, 9, 89.99, 'card', '2024-05-12 08:30:00');

INSERT INTO support_tickets (ticket_id, customer_id, subject, status, priority, created_at, resolved_at) VALUES
(1, 1, 'Late delivery', 'resolved', 'medium', '2024-01-20 09:00:00', '2024-01-21 15:00:00'),
(2, 3, 'Item damaged on arrival', 'open', 'high', '2024-03-05 11:20:00', NULL),
(3, 5, 'Refund request', 'resolved', 'high', '2024-04-05 10:00:00', '2024-04-06 12:00:00'),
(4, 2, 'Question about product', 'closed', 'low', '2024-05-13 14:00:00', '2024-05-13 15:30:00');

INSERT INTO support_agents (agent_id, name, department) VALUES
(1, 'Priya Sharma', 'Support'), (2, 'Tom Baker', 'Support'), (3, 'Elena Popov', 'Escalations');

INSERT INTO ticket_assignments (assignment_id, ticket_id, agent_id, assigned_at) VALUES
(1, 1, 1, '2024-01-20 09:05:00'), (2, 2, 2, '2024-03-05 11:25:00'),
(3, 3, 3, '2024-04-05 10:05:00'), (4, 4, 1, '2024-05-13 14:05:00');

INSERT INTO marketing_campaigns (campaign_id, name, channel, start_date, end_date, budget) VALUES
(1, 'Spring Sale', 'google_ads', '2023-01-01', '2023-02-01', 5000.00),
(2, 'Referral Boost', 'referral', '2023-03-01', '2023-12-31', 2000.00),
(3, 'Instagram Push', 'instagram', '2023-06-01', '2023-08-01', 3000.00);

INSERT INTO campaign_conversions (conversion_id, campaign_id, customer_id, converted_at) VALUES
(1, 1, 1, '2023-01-15 12:00:00'), (2, 1, 5, '2023-09-18 09:00:00'),
(3, 2, 2, '2023-03-22 10:00:00'), (4, 2, 7, '2024-02-14 11:00:00'),
(5, 3, 4, '2023-07-01 13:00:00');
"""

def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.executescript(SAMPLE_DATA)
    conn.commit()
    conn.close()
    print(f"Sample database created at: {DB_PATH} (10 tables)")

if __name__ == "__main__":
    main()
