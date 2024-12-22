'use client'

import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Category, GroceriesApi, Product } from '../../../api/GroceriesApi';
import AllProducts from '../../components/AllProducts';

export default function CategoryPage() {
    const { id } = useParams();
    const [products, setProducts] = useState<Product[]>([]);
    const [category, setCategory] = useState<Category | null>(null);
    const [currentPage, setCurrentPage] = useState(1);
    const [productsPerPage] = useState(12);

    useEffect(() => {
        const fetchData = async () => {
            if (id) {
                const productsData = await GroceriesApi.getProductsByCategory(id as string);
                setProducts(productsData);

                const categoriesData = await GroceriesApi.getAllCategories();
                const currentCategory = categoriesData.find(cat => cat.id === id);
                setCategory(currentCategory || null);
            }
        };
        fetchData();
    }, [id]);

    useEffect(() => {
        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    }, [currentPage])


    // Get current products
    const indexOfLastProduct = currentPage * productsPerPage;
    const indexOfFirstProduct = indexOfLastProduct - productsPerPage;
    const currentProducts = products.slice(indexOfFirstProduct, indexOfLastProduct);

    // Change page
    const paginate = (pageNumber: number) => setCurrentPage(pageNumber);

    return (
        <div className="min-h-screen bg-base-100">
            <main className="container mx-auto py-8">
                {category && (
                    <h1 className="text-3xl font-bold mb-6">{category.name}</h1>
                )}
                <AllProducts products={currentProducts} />
                <div className="flex justify-center mt-8">
                    <div className="btn-group">
                        {Array.from({ length: Math.ceil(products.length / productsPerPage) }, (_, i) => (
                            <button
                                key={i}
                                className={`btn ${currentPage === i + 1 ? 'btn-active' : ''}`}
                                onClick={() => paginate(i + 1)}
                            >
                                {i + 1}
                            </button>
                        ))}
                    </div>
                </div>
            </main>
        </div>
    );
}
