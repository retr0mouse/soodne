'use client'

import React, { useContext } from 'react';
import {  useCart } from '../context/CartContext';
import Image from 'next/image';

export default function Cart() {
    const { cartState, removeFromCart, updateQuantity } = useCart();
    const cartItems = cartState.items;

    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold mb-6">Your Cart</h1>
            {cartItems.length === 0 ? (
                <p>Your cart is empty.</p>
            ) : (
                <>
                    <div className="space-y-4">
                        {cartItems.map((item) => (
                            <div key={item.product.id} className="flex items-center space-x-4 border-b border-gray-200 py-4">
                                <Image
                                    src={item.product.image}
                                    alt={item.product.title}
                                    width={80}
                                    height={80}
                                    className="object-cover"
                                />
                                <div className="flex-grow">
                                    <h2 className="text-lg font-semibold">{item.product.title}</h2>
                                    <p className="text-gray-600">${item.product.price.toFixed(2)}</p>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <button
                                        onClick={() => updateQuantity(item.product.id, item.quantity - 1)}
                                        className="btn btn-sm btn-outline"
                                    >
                                        -
                                    </button>
                                    <input  
                                        value={item.quantity} 
                                        onChange={(e) => {
                                            const newValue = Math.max(1, parseInt(e.target.value) || 1);
                                            updateQuantity(item.product.id, newValue);
                                        }} 
                                        className="input input-bordered input-sm w-16" 
                                        min="1"
                                    />
                                    <button
                                        onClick={() => updateQuantity(item.product.id, item.quantity + 1)}
                                        className="btn btn-sm btn-outline"
                                    >
                                        +
                                    </button>
                                </div>
                                <button
                                    onClick={() => removeFromCart(item.product. id)}
                                    className="btn btn-sm btn-error"
                                >
                                    Remove
                                </button>
                            </div>
                        ))}
                    </div>
                    <div className="mt-8">
                        <p className="text-xl font-semibold">Total: ${cartItems.reduce((total, item) => total + item.product.price * item.quantity, 0).toFixed(2)}</p>
                        <button className="btn btn-primary mt-4">Proceed to Checkout</button>
                    </div>
                </>
            )}
        </div>
    );
}
