CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TYPE matching_status_enum AS ENUM ('unmatched', 'matched', 'pending');

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS Units (
    unit_id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    conversion_factor DECIMAL(10, 6) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE TRIGGER trg_update_units_updated_at
BEFORE UPDATE ON Units
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE TABLE IF NOT EXISTS Stores (
    store_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    website_url TEXT,
    image_url TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE TRIGGER trg_update_stores_updated_at
BEFORE UPDATE ON Stores
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE TABLE IF NOT EXISTS Categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    parent_id INTEGER,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_parent_category
        FOREIGN KEY (parent_id)
            REFERENCES Categories (category_id)
            ON DELETE SET NULL
            ON UPDATE CASCADE,
    CONSTRAINT uq_category_name_parent UNIQUE (name, parent_id)
);

CREATE TRIGGER trg_update_categories_updated_at
BEFORE UPDATE ON Categories
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE INDEX IF NOT EXISTS idx_categories_parent_id
    ON Categories (parent_id);

CREATE INDEX IF NOT EXISTS idx_categories_name
    ON Categories (name);

CREATE TABLE IF NOT EXISTS Products (
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
            REFERENCES Units (unit_id),
    CONSTRAINT fk_category
        FOREIGN KEY (category_id)
            REFERENCES Categories (category_id)
            ON DELETE SET NULL
            ON UPDATE CASCADE
);

CREATE TRIGGER trg_update_products_updated_at
BEFORE UPDATE ON Products
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE INDEX IF NOT EXISTS idx_products_category_id
    ON Products (category_id);

CREATE INDEX IF NOT EXISTS idx_products_name
    ON Products (name);

CREATE INDEX IF NOT EXISTS idx_products_name_lower
    ON Products (LOWER(name));

CREATE INDEX IF NOT EXISTS idx_products_name_trgm
    ON Products
    USING GIN (name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_products_fulltext
    ON Products
    USING GIN (to_tsvector('english', name || ' ' || COALESCE(description, '')));

CREATE TABLE IF NOT EXISTS ProductStoreData (
    product_store_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    price_per_unit DECIMAL(10, 2) CHECK (price_per_unit >= 0),
    last_updated TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    store_product_name VARCHAR(255),
    store_description TEXT,
    store_image_url TEXT,
    store_weight_value DECIMAL(10, 2) CHECK (store_weight_value >= 0),
    store_unit_id INTEGER,
    additional_attributes JSONB,
    matching_status matching_status_enum DEFAULT 'unmatched',
    last_matched TIMESTAMP WITHOUT TIME ZONE,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_product
        FOREIGN KEY (product_id)
            REFERENCES Products (product_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
    CONSTRAINT fk_store
        FOREIGN KEY (store_id)
            REFERENCES Stores (store_id)
            ON DELETE CASCADE
            ON UPDATE CASCADE,
    CONSTRAINT fk_store_unit
        FOREIGN KEY (store_unit_id)
            REFERENCES Units (unit_id)
);

CREATE TRIGGER trg_update_productstoredata_updated_at
BEFORE UPDATE ON ProductStoreData
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at();

CREATE INDEX IF NOT EXISTS idx_productstoredata_product_id
    ON ProductStoreData (product_id);

CREATE INDEX IF NOT EXISTS idx_productstoredata_store_id
    ON ProductStoreData (store_id);

CREATE INDEX IF NOT EXISTS idx_productstoredata_last_updated
    ON ProductStoreData (last_updated);

CREATE INDEX IF NOT EXISTS idx_productstoredata_product_price
    ON ProductStoreData (product_id, price);

CREATE INDEX IF NOT EXISTS idx_productstoredata_price_per_unit
    ON ProductStoreData (price_per_unit);

CREATE INDEX IF NOT EXISTS idx_productstoredata_additional_attributes
    ON ProductStoreData
    USING GIN (additional_attributes);

CREATE INDEX IF NOT EXISTS idx_productstoredata_matching_status
    ON ProductStoreData (matching_status);

CREATE INDEX IF NOT EXISTS idx_productstoredata_store_product_name_trgm
    ON ProductStoreData
    USING GIN (store_product_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_productstoredata_store_description_trgm
    ON ProductStoreData
    USING GIN (store_description gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_stores_name
    ON Stores (name);

CREATE OR REPLACE FUNCTION update_last_updated()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_last_updated
BEFORE UPDATE ON ProductStoreData
FOR EACH ROW
EXECUTE PROCEDURE update_last_updated();

CREATE OR REPLACE FUNCTION reset_matching_status()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.store_product_name <> OLD.store_product_name
        OR NEW.store_description <> OLD.store_description
        OR NEW.store_image_url <> OLD.store_image_url THEN
        NEW.matching_status = 'unmatched';
        NEW.product_id = NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_reset_matching_status
BEFORE UPDATE ON ProductStoreData
FOR EACH ROW
EXECUTE PROCEDURE reset_matching_status();

CREATE TABLE IF NOT EXISTS ProductPriceHistory (
    price_history_id SERIAL PRIMARY KEY,
    product_store_id INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    recorded_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_product_store
        FOREIGN KEY (product_store_id)
            REFERENCES ProductStoreData (product_store_id)
            ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pricehistory_product_store_id
    ON ProductPriceHistory (product_store_id);

CREATE INDEX IF NOT EXISTS idx_pricehistory_recorded_at
    ON ProductPriceHistory (recorded_at);

CREATE OR REPLACE FUNCTION record_price_history()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.price <> OLD.price THEN
        INSERT INTO ProductPriceHistory (product_store_id, price, recorded_at)
        VALUES (NEW.product_store_id, NEW.price, NOW());
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_record_price_history
AFTER UPDATE OF price ON ProductStoreData
FOR EACH ROW
WHEN (OLD.price IS DISTINCT FROM NEW.price)
EXECUTE PROCEDURE record_price_history();

CREATE TABLE IF NOT EXISTS ProductMatchingLog (
    log_id SERIAL PRIMARY KEY,
    product_store_id INTEGER NOT NULL,
    product_id INTEGER,
    confidence_score DECIMAL(5, 2),
    matched_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    matched_by VARCHAR(50),
    CONSTRAINT fk_product_store_log
        FOREIGN KEY (product_store_id)
            REFERENCES ProductStoreData (product_store_id),
    CONSTRAINT fk_product_log
        FOREIGN KEY (product_id)
            REFERENCES Products (product_id)
);

CREATE INDEX IF NOT EXISTS idx_product_matching_log_product_id
    ON ProductMatchingLog (product_id);

CREATE INDEX IF NOT EXISTS idx_product_matching_log_confidence_score
    ON ProductMatchingLog (confidence_score);
