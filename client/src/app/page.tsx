'use client'
import ThemeToggle from '@/app/components/ThemeToggle';
import ProfilePic from './images/mouse.png';
import Image from 'next/image';
import AllCategories from './components/AllCategories';
import { Product, Category, GroceriesApi } from '../../api/GroceriesApi';
import { useState, useEffect, useCallback } from 'react';

export default function Home() {
    const [cartItems, setCartItems] = useState<Product[]>([]);
    const [categories, setCategories] = useState<Category[]>([]);

    useEffect(() => {
        const fetchCategories = async () => {
            const data = await GroceriesApi.getAllCategories();
            setCategories(data);
        };
        fetchCategories();
    }, []);

    const handleAddToCart = useCallback((product: Product) => {
        setCartItems((prevCartItems) => [...prevCartItems, product]);
    }, []);

    return (
        <div className="min-h-screen bg-base-100">
            <main className="container mx-auto py-8">
                <h1 className="text-3xl font-bold mb-6">All Categories</h1>
                <AllCategories categories={categories} />
            </main>
        </div>
    );
}
