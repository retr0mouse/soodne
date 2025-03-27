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
    conversion_factor DECIMAL(10, 6) NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    parent_id INTEGER,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    url VARCHAR(255),
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
    image_url TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    image_url TEXT,
    weight_value DECIMAL(10, 2) CHECK (weight_value >= 0),
    unit_id INTEGER,
    category_id INTEGER,
    barcode VARCHAR(50) UNIQUE,
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

CREATE TABLE IF NOT EXISTS product_store_data (
    product_store_id SERIAL PRIMARY KEY,
    product_id INTEGER,
    store_id INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    price_per_unit DECIMAL(10, 2) CHECK (price_per_unit >= 0),
    last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    store_product_name VARCHAR(255),
    store_image_url TEXT,
    store_product_url TEXT,
    store_category_id INTEGER,
    store_weight_value DECIMAL(10, 2) CHECK (store_weight_value >= 0),
    store_unit_id INTEGER,
    ean VARCHAR(13), 
    additional_attributes JSONB,
    last_matched TIMESTAMP WITHOUT TIME ZONE,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_product
        FOREIGN KEY (product_id)
            REFERENCES products (product_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
    CONSTRAINT fk_store
        FOREIGN KEY (store_id)
            REFERENCES stores (store_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
    CONSTRAINT fk_store_unit
        FOREIGN KEY (store_unit_id)
            REFERENCES units (unit_id),
    CONSTRAINT fk_store_category
        FOREIGN KEY (store_category_id)
            REFERENCES categories (category_id)
            ON DELETE SET NULL
            ON UPDATE CASCADE
);

CREATE TRIGGER trg_update_productstoredata_updated_at
BEFORE UPDATE ON product_store_data
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE INDEX IF NOT EXISTS idx_productstoredata_product_id
    ON product_store_data (product_id);

CREATE INDEX IF NOT EXISTS idx_productstoredata_store_id
    ON product_store_data (store_id);

CREATE INDEX IF NOT EXISTS idx_productstoredata_last_updated
    ON product_store_data (last_updated);

CREATE INDEX IF NOT EXISTS idx_productstoredata_product_price
    ON product_store_data (product_id, price);

CREATE INDEX IF NOT EXISTS idx_productstoredata_price_per_unit
    ON product_store_data (price_per_unit);

CREATE INDEX IF NOT EXISTS idx_productstoredata_additional_attributes
    ON product_store_data
    USING GIN (additional_attributes);

CREATE INDEX IF NOT EXISTS idx_productstoredata_store_product_name_trgm
    ON product_store_data
    USING GIN (store_product_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_productstoredata_ean
    ON product_store_data (ean);

CREATE INDEX IF NOT EXISTS idx_productstoredata_store_product_url
    ON product_store_data (store_product_url);

CREATE INDEX IF NOT EXISTS idx_productstoredata_store_category_id
    ON product_store_data (store_category_id);

CREATE INDEX IF NOT EXISTS idx_stores_name
    ON stores (name);

CREATE OR REPLACE FUNCTION update_last_updated()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_last_updated
BEFORE UPDATE ON product_store_data
FOR EACH ROW
EXECUTE PROCEDURE update_last_updated();

CREATE OR REPLACE FUNCTION reset_matching_status()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.store_product_name <> OLD.store_product_name
        OR NEW.store_image_url <> OLD.store_image_url
        OR NEW.store_product_url <> OLD.store_product_url
        OR NEW.store_category_id <> OLD.store_category_id
        OR NEW.ean <> OLD.ean
        OR NEW.store_weight_value <> OLD.store_weight_value
        OR NEW.store_unit_id <> OLD.store_unit_id THEN
        NEW.product_id = NULL;
        NEW.last_matched = NULL;
    END IF;
    RETURN NEW;
END;    
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_reset_matching_status
BEFORE UPDATE ON product_store_data
FOR EACH ROW
EXECUTE PROCEDURE reset_matching_status();

CREATE TABLE IF NOT EXISTS product_price_history (
    price_history_id SERIAL PRIMARY KEY,
    product_store_id INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    recorded_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_product_store
        FOREIGN KEY (product_store_id)
            REFERENCES product_store_data (product_store_id)
            ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pricehistory_product_store_id
    ON product_price_history (product_store_id);

CREATE INDEX IF NOT EXISTS idx_pricehistory_recorded_at
    ON product_price_history (recorded_at);

CREATE OR REPLACE FUNCTION record_price_history()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.price <> OLD.price THEN
        INSERT INTO product_price_history (product_store_id, price, recorded_at)
        VALUES (NEW.product_store_id, NEW.price, NOW());
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_record_price_history
AFTER UPDATE OF price ON product_store_data
FOR EACH ROW
WHEN (OLD.price IS DISTINCT FROM NEW.price)
EXECUTE PROCEDURE record_price_history();

CREATE TABLE IF NOT EXISTS product_matching_log (
    log_id SERIAL PRIMARY KEY,
    product_store_id INTEGER NOT NULL,
    product_id INTEGER,
    confidence_score DECIMAL(5, 2),
    matched_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    matched_by VARCHAR(50),
    CONSTRAINT fk_product_store_log
        FOREIGN KEY (product_store_id)
            REFERENCES product_store_data (product_store_id),
    CONSTRAINT fk_product_log
        FOREIGN KEY (product_id)
            REFERENCES products (product_id)
);

CREATE INDEX IF NOT EXISTS idx_product_matching_log_product_id
    ON product_matching_log (product_id);

CREATE INDEX IF NOT EXISTS idx_product_matching_log_confidence_score
    ON product_matching_log (confidence_score);

CREATE INDEX IF NOT EXISTS idx_productstoredata_unmatched
    ON product_store_data (product_id)
    WHERE product_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_productstoredata_potential_matches
    ON product_store_data (store_id, product_id)
    WHERE product_id IS NOT NULL;
