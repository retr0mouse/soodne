'use client'

import React, { createContext, useContext, useState, ReactNode } from 'react';
import { Product } from '../../api/GroceriesApi';

interface CartItem {
    product: Product;
    quantity: number;
}

interface CartState {
    items: CartItem[];
}

interface CartContextType {
    cartState: CartState;
    addToCart: (product: Product) => void;
    removeFromCart: (productId: string) => void;
    updateQuantity: (productId: string, quantity: number) => void;
}

const CartContext = createContext<CartContextType | undefined>(undefined);

export const CartProvider = ({ children }: { children: ReactNode }) => {
    const [cartState, setCartState] = useState<CartState>({ items: [] });

    const addToCart = (product: Product) => {
        setCartState((prevState) => {
            const existingItem = prevState.items.find(item => item.product.id === product.id);
            if (existingItem) {
                return {
                    ...prevState,
                    items: prevState.items.map(item =>
                        item.product.id === product.id
                            ? { ...item, quantity: item.quantity + 1 }
                            : item
                    )
                };
            } else {
                return {
                    ...prevState,
                    items: [...prevState.items, { product, quantity: 1 }]
                };
            }
        });
    };

    const removeFromCart = (productId: string) => {
        setCartState((prevState) => ({
            ...prevState,
            items: prevState.items.filter(item => item.product.id !== productId)
        }));
    };

    const updateQuantity = (productId: string, quantity: number) => {
        setCartState((prevState) => ({
            ...prevState,
            items: prevState.items.map(item =>
                item.product.id === productId
                    ? { ...item, quantity: Math.max(0, quantity) }
                    : item
            ).filter(item => item.quantity > 0)
        }));
    };

    return (
        <CartContext.Provider value={{ cartState, addToCart, removeFromCart, updateQuantity }}>
            {children}
        </CartContext.Provider>
    );
};

export const useCart = () => {
    const context = useContext(CartContext);
    if (!context) {
        throw new Error('useCart must be used within a CartProvider');
    }
    return context;
};
