'use client'

import AllCategories from './components/AllCategories';
import { GroceriesApi } from '../api/GroceriesApi';
import { useState, useEffect, useCallback } from 'react';
import {Product} from "../types/ProductTypes";
import AllProducts from "./components/AllProducts";

export default function Home() {
    const [cartItems, setCartItems] = useState<Product[]>([]);
    const [allProducts, setAllProducts] = useState<Product[]>([])
    // const [categories, setCategories] = useState<Category[]>([]);

    useEffect(() => {
        // const fetchCategories = async () => {
        //     const data = await GroceriesApi.getAllCategories();
        //     setCategories(data);
        // };
        // fetchCategories();
        const fetchAllProducts = async () => {
            const data = await GroceriesApi.getAllProducts();
            setAllProducts(data);
        };
        fetchAllProducts();

    }, []);

    const handleAddToCart = useCallback((product: Product) => {
        setCartItems((prevCartItems) => [...prevCartItems, product]);
    }, []);

    return (
        <div className="min-h-screen bg-base-100">
            <main className="container mx-auto py-8">
                <h1 className="text-3xl font-bold mb-6">All Products</h1>
                <AllProducts products={allProducts} />
            </main>
        </div>
    );
}