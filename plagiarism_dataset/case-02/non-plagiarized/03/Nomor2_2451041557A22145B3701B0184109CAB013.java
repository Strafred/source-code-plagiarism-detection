import java.util.Scanner;
public class Nomor2_2451041557A22145B3701B0184109CAB013 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder : ");
        double radius = sc.nextDouble();
        double length = sc.nextDouble();
        double area = (radius * radius) * 3.14159;
        double volume = area * length;
        System.out.println("The area is : " + area);
        System.out.println("The volume is : " + volume);
    }
}
