/*****************************************************************//**
 * \file   SparseMatrix.cpp
 * \brief  稀疏矩阵数学计算方法
 * 
 * \author LUOJIAXUAN
 * \date   May 22nd 2024
 *********************************************************************/
#include "SparseMatrix.h"

template<class T>
void SparseSurfelFusion::SparseMatrix<T>::Resize(int r)
{
    int i;
    if (rows > 0) {
        free(m_ppElements);
        free(rowSizes);
    }
    rows = r;
    if (r) {
        rowSizes = (int*)malloc(sizeof(int) * r);
        memset(rowSizes, 0, sizeof(int) * r);
        m_ppElements = (MatrixEntry<T>**)malloc(sizeof(MatrixEntry<T>*) * r);
    }
}

template<class T>
void SparseSurfelFusion::SparseMatrix<T>::SetRowSize(int row, int count)
{
    if (row >= 0 && row < rows) {
        if (rowSizes[row]) { free(m_ppElements[row]); }
        if (count > 0) { m_ppElements[row] = (MatrixEntry<T>*)malloc(sizeof(MatrixEntry<T>) * count); }
        rowSizes[row] = count;
    }
}

template<class T>
int SparseSurfelFusion::SparseMatrix<T>::Entries(void)
{
    int e = 0;
    for (int i = 0; i < rows; i++) { e += int(rowSizes[i]); }
    return e;
}

template<class T>
void SparseSurfelFusion::SparseMatrix<T>::SetZero()
{
    Resize(rows);
}

template<class T>
void SparseSurfelFusion::SparseMatrix<T>::SetIdentity()
{
    SetZero();
    //    for(int ij=0; ij < Min( this->Rows(), this->Columns() ); ij++)
    //        (*this)(ij,ij) = T(1);
    for (int ij = 0; ij < rows; ij++) {
        SetRowSize(ij, 1);
        m_ppElements[ij][0] = MatrixEntry<T>(ij);
        m_ppElements[ij][0].Value = T(1);
    }
}

template<class T>
SparseSurfelFusion::SparseMatrix<T>& SparseSurfelFusion::SparseMatrix<T>::operator=(const SparseMatrix<T>& M)
{
    Resize(M.rows);
    for (int i = 0; i < rows; i++) {
        SetRowSize(i, M.rowSizes[i]);
        for (int j = 0; j < rowSizes[i]; j++) { m_ppElements[i][j] = M.m_ppElements[i][j]; }
    }
    return *this;
}

template<class T>
SparseSurfelFusion::SparseMatrix<T> SparseSurfelFusion::SparseMatrix<T>::operator*(const T& V) const
{
    SparseMatrix<T> M(*this);
    M *= V;
    return M;
}

template<class T>
SparseSurfelFusion::SparseMatrix<T>& SparseSurfelFusion::SparseMatrix<T>::operator*=(const T& V)
{
    for (int i = 0; i < this->Rows(); i++)
    {
        for (int ii = 0; ii < m_ppElements[i].size(); ii++) { m_ppElements[i][ii].Value *= V; }
    }
    return *this;
}

template<class T>
SparseSurfelFusion::SparseMatrix<T> SparseSurfelFusion::SparseMatrix<T>::operator*(const SparseMatrix<T>& M) const
{
    return Multiply(M);
}


template<class T>
template<class T2>
SparseSurfelFusion::Vector<T2> SparseSurfelFusion::SparseMatrix<T>::operator*(const Vector<T2>& V) const
{
    Vector<T2> R(rows);

    for (int i = 0; i < rows; i++)
    {
        T2 temp = T2();
        for (int ii = 0; ii < rowSizes[i]; ii++) {
            temp += m_ppElements[i][ii].Value * V.m_pV[m_ppElements[i][ii].N];
        }
        R(i) = temp;
    }
    return R;
}



template<class T>
SparseSurfelFusion::SparseMatrix<T> SparseSurfelFusion::SparseMatrix<T>::Multiply(const SparseMatrix<T>& M) const
{
    SparseMatrix<T> R(this->Rows(), M.Columns());
    for (int i = 0; i < R.Rows(); i++) {
        for (int ii = 0; ii < m_ppElements[i].size(); ii++) {
            int N = m_ppElements[i][ii].N;
            T Value = m_ppElements[i][ii].Value;
            for (int jj = 0; jj < M.m_ppElements[N].size(); jj++) {
                R(i, M.m_ppElements[N][jj].N) += Value * M.m_ppElements[N][jj].Value;
            }
        }
    }
    return R;
}


template<class T>
template<class T2>
SparseSurfelFusion::Vector<T2> SparseSurfelFusion::SparseMatrix<T>::Multiply(const Vector<T2>& V) const
{
    Vector<T2> R(rows);

    for (int i = 0; i < rows; i++)
    {
        T2 temp = T2();
        for (int ii = 0; ii < rowSizes[i]; ii++) {
            temp += m_ppElements[i][ii].Value * V.m_pV[m_ppElements[i][ii].N];
        }
        R(i) = temp;
    }
    return R;
}

template<class T>
template<class T2>
void SparseSurfelFusion::SparseMatrix<T>::Multiply(const Vector<T2>& In, Vector<T2>& Out) const
{
    for (int i = 0; i < rows; i++) {
        T2 temp = T2();
        for (int j = 0; j < rowSizes[i]; j++) { temp += m_ppElements[i][j].Value * In.m_pV[m_ppElements[i][j].N]; }
        Out.m_pV[i] = temp;
    }
}

template<class T>
SparseSurfelFusion::SparseMatrix<T> SparseSurfelFusion::SparseMatrix<T>::MultiplyTranspose(const SparseMatrix<T>& Mt) const
{

}

template<class T>
SparseSurfelFusion::SparseMatrix<T> SparseSurfelFusion::SparseMatrix<T>::Transpose() const
{
    SparseMatrix<T> M(rows);

    for (int i = 0; i < rows; i++)
    {
        for (int ii = 0; ii < m_ppElements[i].size(); ii++) {
            M(m_ppElements[i][ii].N, i) = m_ppElements[i][ii].Value;
        }
    }
    return M;
}

template<class T>
int SparseSurfelFusion::SparseMatrix<T>::Solve(const SparseMatrix<T>& M, const Vector<T>& b, const int& iters, Vector<T>& solution, const T eps)
{
    SparseMatrix mTranspose = M.Transpose();
    Vector<T> bb = mTranspose * b;
    Vector<T> d, r, Md;
    T alpha, beta, rDotR;
    int i;

    solution.Resize(M.Columns());
    solution.SetZero();

    d = r = bb;
    rDotR = r.Dot(r);

    for (i = 0; i<iters && rDotR>eps; i++) {
        T temp;
        Md = mTranspose * (M * d);
        alpha = rDotR / d.Dot(Md);
        solution += d * alpha;
        r -= Md * alpha;
        temp = r.Dot(r);
        beta = temp / rDotR;
        rDotR = temp;
        d = r + d * beta;
    }
    return i;
}

template<class T>
template<class T2>
int SparseSurfelFusion::SparseMatrix<T>::SolveSymmetric(const SparseMatrix<T>& M, const Vector<T2>& b, const int& iters, Vector<T2>& solution, const T2 eps, const int& reset)
{
    Vector<T2> d, r, Md;
    T2 alpha, beta, rDotR;
    Md.Resize(b.Dimensions());
    if (reset) {
        solution.Resize(b.Dimensions());
        solution.SetZero();
    }
    d = r = b - M.Multiply(solution);
    rDotR = r.Dot(r);
    if (b.Dot(b) <= eps) {
        solution.SetZero();
        return 0;
    }

    int i;
    for (i = 0; i < iters; i++) {
        T2 temp;
        M.Multiply(d, Md);
        temp = d.Dot(Md);
        if (temp <= eps) { break; }
        alpha = rDotR / temp;
        r.SubtractScaled(Md, alpha);
        temp = r.Dot(r);
        if (temp / b.Dot(b) <= eps) { break; }
        beta = temp / rDotR;
        solution.AddScaled(d, alpha);
        if (beta <= eps) { break; }
        rDotR = temp;
        Vector<T2>::Add(d, beta, r, d);
    }
    return i;
}



template<class T, int Dim>
void SparseSurfelFusion::SparseNMatrix<T, Dim>::Resize(int r)
{
    int i;
    if (rows > 0) {
        free(m_ppElements);
        free(rowSizes);
    }
    rows = r;
    if (r) {
        rowSizes = (int*)malloc(sizeof(int) * r);
        memset(rowSizes, 0, sizeof(int) * r);
        m_ppElements = (NMatrixEntry<T, Dim>**)malloc(sizeof(NMatrixEntry<T, Dim>*) * r);
    }
}

template<class T, int Dim>
void SparseSurfelFusion::SparseNMatrix<T, Dim>::SetRowSize(int row, int count)
{
    if (row >= 0 && row < rows) {
        if (rowSizes[row]) { free(m_ppElements[row]); }
        if (count > 0) { m_ppElements[row] = (NMatrixEntry<T, Dim>*)malloc(sizeof(NMatrixEntry<T, Dim>) * count); }
        rowSizes[row] = count;
    }
}

template<class T, int Dim>
int SparseSurfelFusion::SparseNMatrix<T, Dim>::Entries(void)
{
    int e = 0;
    for (int i = 0; i < rows; i++) { e += int(rowSizes[i]); }
    return e;
}

template<class T, int Dim>
SparseSurfelFusion::SparseNMatrix<T, Dim>& SparseSurfelFusion::SparseNMatrix<T, Dim>::operator=(const SparseNMatrix& M)
{
    Resize(M.rows);
    for (int i = 0; i < rows; i++) {
        SetRowSize(i, M.rowSizes[i]);
        for (int j = 0; j < rowSizes[i]; j++) { m_ppElements[i][j] = M.m_ppElements[i][j]; }
    }
    return *this;
}

template<class T, int Dim>
SparseSurfelFusion::SparseNMatrix<T, Dim> SparseSurfelFusion::SparseNMatrix<T, Dim>::operator*(const T& V) const
{
    SparseNMatrix<T, Dim> M(*this);
    M *= V;
    return M;
}

template<class T, int Dim>
SparseSurfelFusion::SparseNMatrix<T, Dim>& SparseSurfelFusion::SparseNMatrix<T, Dim>::operator*=(const T& V)
{
    for (int i = 0; i < rows; i++)
    {
        for (int ii = 0; ii < m_ppElements[i].size(); ii++) {
            for (int jj = 0; jj < Dim; jj++) {
                m_ppElements[i][ii].Value[jj] *= V;
            }
        }
    }
    return *this;
}

template<class T, int Dim>
template<class T2>
SparseSurfelFusion::NVector<T2, Dim> SparseSurfelFusion::SparseNMatrix<T, Dim>::operator*(const Vector<T2>& V) const
{
    NVector<T2, Dim> R(rows);

    for (int i = 0; i < rows; i++)
    {
        T2 temp[Dim];
        for (int ii = 0; ii < Dim; ii++) { temp[ii] = T2(); }
        for (int ii = 0; ii < rowSizes[i]; ii++) {
            for (int jj = 0; jj < Dim; jj++) { temp[jj] += m_ppElements[i][ii].Value[jj] * V.m_pV[m_ppElements[i][jj].N]; }
        }
        for (int ii = 0; ii < Dim; ii++) { R[i][ii] = temp[ii]; }
    }
    return R;
}

template<class T, int Dim>
template<class T2>
SparseSurfelFusion::Vector<T2> SparseSurfelFusion::SparseNMatrix<T, Dim>::operator*(const NVector<T2, Dim>& V) const
{
    Vector<T2> R(rows);

    for (int i = 0; i < rows; i++)
    {
        T2 temp(0);
        for (int ii = 0; ii < rowSizes[i]; ii++) {
            for (int jj = 0; jj < Dim; jj++) { temp += m_ppElements[i][ii].Value[jj] * V.m_pV[m_ppElements[i][ii].N][jj]; }
        }
        R(i) = temp;
    }
    return R;
}

template<class T>
template<class T2>
SparseSurfelFusion::Vector<T2> SparseSurfelFusion::SparseSymmetricMatrix<T>::Multiply(const Vector<T2>& V) const
{
    Vector<T2> R(this->rows);
    for (int i = 0; i < this->rows; i++) {
        for (int ii = 0; ii < this->rowSizes[i]; ii++) {
            int j = this->m_ppElements[i][ii].N;
            R(i) += this->m_ppElements[i][ii].Value * V.m_pV[j];
            R(j) += this->m_ppElements[i][ii].Value * V.m_pV[i];
        }
    }
    return R;
}

template<class T>
template<class T2>
void SparseSurfelFusion::SparseSymmetricMatrix<T>::Multiply(const Vector<T2>& In, Vector<T2>& Out) const
{
    Out.SetZero();
    for (int i = 0; i < this->rows; i++) {
        MatrixEntry<T>* temp = this->m_ppElements[i];
        T2& in1 = In.m_pV[i];
        T2& out1 = Out.m_pV[i];
        int rs = this->rowSizes[i];
        for (int ii = 0; ii < rs; ii++) {
            MatrixEntry<T>& temp2 = temp[ii];
            int j = temp2.N;
            T2 v = temp2.Value;
            out1 += v * In.m_pV[j];
            Out.m_pV[j] += v * in1;
        }
    }
}

template<class T>
template<class T2>
int SparseSurfelFusion::SparseSymmetricMatrix<T>::Solve(const SparseSymmetricMatrix<T>& M, const Vector<T2>& b, const int& iters, Vector<T2>& solution, const T2 eps, const int& reset)
{
    Vector<T2> d, r, Md;
    T2 alpha, beta, rDotR, bDotB;
    Md.Resize(b.Dimensions());
    if (reset) {
        solution.Resize(b.Dimensions());
        solution.SetZero();
    }
    d = r = b - M.Multiply(solution);     // error vector
    rDotR = r.Dot(r);                 // L2 distance of error vector
    bDotB = b.Dot(b);                 // L2 distance of b
    if (b.Dot(b) <= eps) {
        solution.SetZero();
        return 0;
    }
    int i;
    for (i = 0; i < iters; i++) {
        T2 temp;
        M.Multiply(d, Md);           // vec Md = matrix M * vec d
        temp = d.Dot(Md);
        if (fabs(temp) <= eps) { break; }
        alpha = rDotR / temp;
        r.SubtractScaled(Md, alpha);
        temp = r.Dot(r);
        if (temp / bDotB <= eps) { break; }
        beta = temp / rDotR;
        solution.AddScaled(d, alpha);
        if (beta <= eps) { break; }
        rDotR = temp;
        Vector<T2>::Add(d, beta, r, d);
    }
    return i;
}

template<class T>
template<class T2>
int SparseSurfelFusion::SparseSymmetricMatrix<T>::Solve(const SparseSymmetricMatrix<T>& M, const Vector<T>& diagonal, const Vector<T2>& b, const int& iters, Vector<T2>& solution, const T2 eps, const int& reset)
{
    Vector<T2> d, r, Md;
    if (reset) {
        solution.Resize(b.Dimensions());
        solution.SetZero();
    }
    Md.Resize(M.rows);
    for (int i = 0; i < iters; i++) {
        M.Multiply(solution, Md);
        r = b - Md;
        for (int j = 0; j<int(M.rows); j++) { solution[j] += r[j] / diagonal[j]; }
    }
    return iters;
}
