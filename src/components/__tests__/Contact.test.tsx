import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Contact from '../Contact.tsx';

describe('Contact', () => {
  it('renders heading "LET\'S FIX YOUR DATA"', () => {
    render(<Contact />);
    expect(screen.getByText("LET'S FIX YOUR DATA")).toBeInTheDocument();
  });

  it('renders form placeholders from data', () => {
    render(<Contact />);
    expect(screen.getByPlaceholderText('you@company.com')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Describe your data pain point...')).toBeInTheDocument();
  });

  it('renders contact info items', () => {
    render(<Contact />);
    expect(screen.getByText('Global Presence')).toBeInTheDocument();
    expect(screen.getByText('System Status')).toBeInTheDocument();
    expect(screen.getByText('Remote-First, Worldwide')).toBeInTheDocument();
  });

  it('renders submit button "GET IN TOUCH"', () => {
    render(<Contact />);
    expect(screen.getByText('GET IN TOUCH')).toBeInTheDocument();
  });
});
