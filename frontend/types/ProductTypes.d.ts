export interface Product {
    product_id: number;
    name: string;
    description: string | null;
    image_url: string | null;
    weight_value: number | null;
    unit_id: number | null;
    category_id: number | null;
    barcode: string | null;
    created_at: string;
    updated_at: string | null;
    store_data: StoreData[];
}

export interface StoreData {
    product_store_id: number;
    store_id: number;
    price: number;
    price_per_unit: number | null;
    store_product_name: string;
    store_image_url: string | null;
    store_weight_value: number | null;
    store_unit_id: number;
    ean: string | null;
    last_updated: string;
    created_at: string;
    updated_at: string | null;
}
