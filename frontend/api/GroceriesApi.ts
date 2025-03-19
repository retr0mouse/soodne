import { Product, StoreData } from "../types/ProductTypes";

export class GroceriesApi {
    static async getAllProducts(): Promise<Product[]> {
        const data = await fetch('http://localhost:8000/api/v1/products/with-prices/');
        return data.json();
    }

    // static async getAllCategories(): Promise<Category[]> {
    //     const data = await fetch('http://localhost:5050/allCategories');
    //     return data.json();
    // }
    //
    // static async getProductsByCategory(categoryId: string): Promise<Product[]> {
    //     const data = await fetch(`http://localhost:5050/productsByCategory/${categoryId}`);
    //     return data.json();
    // }
}

