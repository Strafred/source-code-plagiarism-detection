import java.util.Scanner;
public class Level6 
{
    static Scanner sc = new Scanner(System.in);
    public static void tampil(int[]angka){
        for (int x = 10; x > 0; x--)
            System.out.println(angka[x]);
    }
    public static void main(String[] args) 
    {
        int arrAngka[] = new int[10];
        for (int x = 0; x < arrAngka.length; x++) 
        {
            System.out.print("Read a number: ");
            arrAngka[x] = sc.nextInt();
        }
        tampil(arrAngka);
    }
}
