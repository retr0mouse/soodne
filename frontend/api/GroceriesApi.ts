export interface Product {
    id: string;
    title: string;
    description: string;
    price: number;
    price_per_unit: number;
    unit_of_measure: string;
    image: string;
}

export interface Category {
    id: string;
    name: string;
    image: string;
    description: string;
}


export class GroceriesApi {
    static async getAllProducts(): Promise<Product[]> {
        const data = await fetch('http://localhost:5050/allProducts');
        return data.json();
    }

    static async getAllCategories(): Promise<Category[]> {
        const data = await fetch('http://localhost:5050/allCategories');
        return data.json();
    }

    static async getProductsByCategory(categoryId: string): Promise<Product[]> {
        const data = await fetch(`http://localhost:5050/productsByCategory/${categoryId}`);
        return data.json();
    }
}

