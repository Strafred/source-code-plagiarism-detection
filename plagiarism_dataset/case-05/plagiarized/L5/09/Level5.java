import java.util.Scanner;
public class Level5 {
    public static void main(String[] args) {
        Scanner masukkan = new Scanner(System.in);
        int angka, sisaBagi;
        System.out.print("Enter an integer: ");
        angka = masukkan.nextInt();
        while (angka != 0) { 
            sisaBagi = angka % 10;
            System.out.print(sisaBagi);
            angka = angka / 10;
        }
        System.out.println();
    }
}
