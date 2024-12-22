'use client'
import React, { ReactElement } from 'react';
import { useCart } from '../context/CartContext';
import Link from 'next/link';

export default function Cart(): ReactElement {
    const { cartState, removeFromCart, updateQuantity } = useCart();
    const cartItems = cartState.items;
    const [isOpen, setIsOpen] = React.useState(false);

    const toggleCart = () => setIsOpen(!isOpen);

    const cartCount = cartItems.reduce((total, item) => total + item.quantity, 0);
    const cartSubtotal = cartItems.reduce((total, item) => total + item.product.price * item.quantity, 0);

    return (
        <div className="dropdown dropdown-end">
            <div tabIndex={0} role="button" className="btn btn-ghost btn-circle" onClick={toggleCart}>
                <div className="indicator">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-5 w-5"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
                    </svg>
                    <span className="badge badge-sm indicator-item">{cartCount}</span>
                </div>
            </div>
            {isOpen && (
                <div
                    className="card card-compact dropdown-content bg-base-100 z-[1] mt-3 w-80 shadow">
                    <div className="card-body">
                        <span className="text-lg font-bold">{cartCount} Item{cartCount !== 1 ? 's' : ''}</span>
                        {cartItems.map((item, index) => (
                            <div key={index} className="flex justify-between items-center mb-2">
                                <span className="flex-grow">{item.product.title}</span>
                                <span className="mx-2">${(item.product.price * item.quantity).toFixed(2)}</span>
                                <div className="flex items-center">
                                    <button className="btn btn-xs btn-circle btn-outline" onClick={() => updateQuantity(item.product.id, item.quantity - 1)}>-</button>
                                    <span className="mx-2">{item.quantity}</span>
                                    <button className="btn btn-xs btn-circle btn-outline" onClick={() => updateQuantity(item.product.id, item.quantity + 1)}>+</button>
                                    <button className="btn btn-xs btn-circle btn-error ml-2" onClick={() => removeFromCart(item.product.id)}>×</button>
                                </div>
                            </div>
                        ))}
                        <span className="text-info font-semibold mt-2">Subtotal: ${cartSubtotal.toFixed(2)}</span>
                        <div className="card-actions">
                            <Link href="/cart" className="btn btn-primary btn-block">View cart</Link>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
