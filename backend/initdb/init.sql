CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TYPE matching_status_enum AS ENUM ('unmatched', 'matched', 'pending');

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS units (
    unit_id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    conversion_factor DECIMAL(10, 6) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE TRIGGER trg_update_units_updated_at
BEFORE UPDATE ON units
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE TABLE IF NOT EXISTS categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    parent_id INTEGER,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_parent_category
        FOREIGN KEY (parent_id)
            REFERENCES categories (category_id)
            ON DELETE SET NULL
            ON UPDATE CASCADE,
    CONSTRAINT uq_category_name_parent UNIQUE (name, parent_id)
);

CREATE TRIGGER trg_update_categories_updated_at
BEFORE UPDATE ON categories
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE INDEX IF NOT EXISTS idx_categories_parent_id
    ON categories (parent_id);

CREATE INDEX IF NOT EXISTS idx_categories_name
    ON categories (name);

CREATE TABLE IF NOT EXISTS stores (
    store_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    website_url TEXT,
    image_url TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE TRIGGER trg_update_stores_updated_at
BEFORE UPDATE ON stores
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    image_url TEXT,
    weight_value DECIMAL(10, 2) CHECK (weight_value >= 0),
    unit_id INTEGER,
    category_id INTEGER,
    barcode VARCHAR(50) UNIQUE,
    matching_status matching_status_enum DEFAULT 'unmatched',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_unit
        FOREIGN KEY (unit_id)
            REFERENCES units (unit_id),
    CONSTRAINT fk_category
        FOREIGN KEY (category_id)
            REFERENCES categories (category_id)
            ON DELETE SET NULL
            ON UPDATE CASCADE
);

CREATE TRIGGER trg_update_products_updated_at
BEFORE UPDATE ON products
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE INDEX IF NOT EXISTS idx_products_category_id
    ON products (category_id);

CREATE INDEX IF NOT EXISTS idx_products_name
    ON products (name);

CREATE INDEX IF NOT EXISTS idx_products_name_lower
    ON products (LOWER(name));

CREATE INDEX IF NOT EXISTS idx_products_name_trgm
    ON products
    USING GIN (name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_products_fulltext
    ON products
    USING GIN (to_tsvector('english', name || ' ' || COALESCE(description, '')));

CREATE TABLE IF NOT EXISTS product_prices (
    price_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_product
        FOREIGN KEY (product_id)
            REFERENCES products (product_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
    CONSTRAINT fk_store
        FOREIGN KEY (store_id)
            REFERENCES stores (store_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_product_prices_product_id
    ON product_prices (product_id);

CREATE INDEX IF NOT EXISTS idx_product_prices_store_id
    ON product_prices (store_id);

CREATE INDEX IF NOT EXISTS idx_product_prices_created_at
    ON product_prices (created_at);

CREATE TABLE IF NOT EXISTS product_matching_log (
    log_id SERIAL PRIMARY KEY,
    product_id1 INTEGER NOT NULL,
    product_id2 INTEGER NOT NULL,
    confidence_score DECIMAL(5, 2),
    matched_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    matched_by VARCHAR(50),
    CONSTRAINT fk_product1_log
        FOREIGN KEY (product_id1)
            REFERENCES products (product_id)
            ON DELETE CASCADE,
    CONSTRAINT fk_product2_log
        FOREIGN KEY (product_id2)
            REFERENCES products (product_id)
            ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_product_matching_log_product_ids
    ON product_matching_log (product_id1, product_id2);

CREATE INDEX IF NOT EXISTS idx_product_matching_log_confidence_score
    ON product_matching_log (confidence_score);
